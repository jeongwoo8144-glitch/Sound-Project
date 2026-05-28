"""Custom DNN Classifier training on YAMNet embeddings.
(YAMNet 임베딩 위에서 학습하는 맞춤형 DNN 분류기)

Step 4 of the ADAS Sound Detector pipeline.
(ADAS 음향 탐지기 파이프라인의 4번째 단계)

Architecture (configurable via config.yaml):
    Input (1024-d YAMNet embedding)
    → Dense(256, ReLU) + BatchNorm + Dropout(0.3)
    → Dense(64,  ReLU) + Dropout(0.3)
    → Dense(num_classes, Softmax)

개선사항:
  - num_classes는 config 입력 없이 데이터에서 완전 자동 결정됩니다.
  - 클래스 이름 표시가 config.yaml의 target_classes에서 자동으로 가져와집니다.
  - load_config / setup_logging 중복 제거 → utils.config에서 import.

[ 모델 학습 전략 ]
  YAMNet은 완전히 동결(frozen)하고, 상단 분류기 헤드만 학습합니다.
  YAMNet이 AudioSet(200만+ 오디오 샘플)에서 학습한 임베딩을
  ADAS 도메인(경적/사이렌/배경)에 맞게 분류하는 역할입니다.

  클래스 정의:
    car_horn  (class_id=1)  → dense label 0
    siren     (class_id=8)  → dense label 1
    background(class_id=99) → dense label 2
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

# Windows에서 sklearn보다 TF를 먼저 import해야 DLL 충돌 방지
import tensorflow as tf  # type: ignore[import]
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from .utils.config import get_class_map, load_config, setup_logging


# ---------------------------------------------------------------------------
# 데이터 로딩 + 폴드 분할
# ---------------------------------------------------------------------------

def load_embedding_dataset(
    npz_path: Path,
    manifest_path: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    """YAMNet 임베딩 파일(embeddings.npz)을 로드하고 manifest CSV와 결합합니다.

    [ 동작 원리 ]
    embeddings.npz: YAMNet이 추출한 1024차원 임베딩 벡터 (embedding.py에서 생성)
    manifest.csv:   각 오디오 클립의 메타데이터 (폴드 번호, 클래스, SNR 등)
    두 파일을 파일 경로(path) 기준으로 inner join하여 하나의 데이터프레임으로 합칩니다.

    Args:
        npz_path: embedding.py가 생성한 embeddings.npz 경로
        manifest_path: data_pipeline.py가 생성한 manifest.csv 경로
        logger: 로깅 인스턴스

    Returns:
        [path, fold, class_id, class, snr_db, original_file, embedding] 컬럼 포함 DataFrame

    Raises:
        FileNotFoundError: 파일이 없는 경우
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"embeddings.npz not found: {npz_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.csv not found: {manifest_path}")

    # NPZ 파일에서 임베딩, 레이블, 경로 로드
    data = np.load(str(npz_path), allow_pickle=True)
    X: np.ndarray = data["X"]          # shape: (N, 1024) — 1024차원 임베딩
    y: np.ndarray = data["y"]          # shape: (N,) — 원본 class_id (1, 8, 99)
    paths: np.ndarray = data["paths"].astype(str)

    # 경로를 기준으로 임베딩 DataFrame 생성
    emb_df = pd.DataFrame({"path": paths, "class_id_npz": y})

    # manifest CSV 로드 및 경로 구분자 통일 (Windows '\\' → '/')
    manifest_df = pd.read_csv(str(manifest_path))
    manifest_df["path"] = manifest_df["path"].str.replace("\\", "/", regex=False)
    emb_df["path"] = emb_df["path"].str.replace("\\", "/", regex=False)

    # path 기준 inner join → 메타데이터 결합
    merged = emb_df.merge(manifest_df, on="path", how="inner")
    if len(merged) != len(emb_df):
        # 경로 불일치로 드롭된 샘플이 있으면 경고
        logger.warning(
            "Join dropped %d rows (path mismatch between npz and manifest).",
            len(emb_df) - len(merged),
        )

    # 경로 → 인덱스 매핑으로 각 행에 해당하는 임베딩 벡터 연결
    path_to_idx = {p: i for i, p in enumerate(emb_df["path"])}
    emb_indices = merged["path"].map(path_to_idx).values
    merged["embedding"] = list(X[emb_indices])   # 각 행에 1024차원 벡터 첨부

    logger.info("Dataset loaded: %d samples, %d unique folds", len(merged), merged["fold"].nunique())
    return merged


def build_label_map(df: pd.DataFrame) -> dict[int, int]:
    """희소(sparse) class_id를 연속된 dense label (0, 1, …)로 매핑합니다.

    [ 이유 ]
    UrbanSound8K의 class_id는 1, 8, 99처럼 불연속적입니다.
    Keras의 SparseCategoricalCrossentropy는 0부터 시작하는 연속 정수를 요구하므로
    매핑이 필요합니다.

    [ 매핑 규칙 ]
    - 양성(안전 위험) 클래스를 먼저 배치: car_horn(1→0), siren(8→1)
    - background(99)는 맨 마지막: 99→2
    - config 수동 입력 없이 데이터에서 자동 결정

    예시: {1: 0, 8: 1, 99: 2}
    """
    BACKGROUND_ID = 99
    unique_ids = sorted(df["class_id"].unique())

    # background(99)를 맨 뒤로 이동시켜 양성 클래스가 앞에 오도록
    if BACKGROUND_ID in unique_ids:
        unique_ids = [i for i in unique_ids if i != BACKGROUND_ID] + [BACKGROUND_ID]

    return {cid: idx for idx, cid in enumerate(unique_ids)}


def verify_no_leakage(df: pd.DataFrame, logger: logging.Logger) -> None:
    """데이터 누수(Data Leakage) 검사: 하나의 원본 파일이 여러 폴드에 걸치지 않는지 확인.

    [ 데이터 누수란? ]
    UrbanSound8K에서 하나의 원본 녹음을 여러 4초 클립으로 나눴을 때,
    같은 원본의 클립이 train과 test에 동시에 들어가면 정보 누출(leakage)이 발생합니다.
    → 테스트 성능이 실제보다 부풀려짐

    UrbanSound8K는 원래 fold 단위로 파일을 분리했으므로 이 검사가 통과되어야 합니다.

    Args:
        df: original_file, fold 컬럼을 가진 DataFrame
        logger: 로깅 인스턴스

    Raises:
        RuntimeError: 한 원본 파일이 여러 폴드에 걸친 경우
    """
    fold_per_file = df.groupby("original_file")["fold"].nunique()
    leaky = fold_per_file[fold_per_file > 1]
    if len(leaky) > 0:
        raise RuntimeError(
            f"Data leakage detected! {len(leaky)} source files span multiple folds:\n"
            f"{leaky.index.tolist()[:10]}"
        )
    logger.info(
        "Leakage check PASSED — %d unique source files, all confined to a single fold.",
        len(fold_per_file),
    )


def fold_split(
    df: pd.DataFrame,
    val_folds: list[int],
    test_folds: list[int],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """폴드 번호 기준으로 데이터를 train / val / test로 분할합니다.

    [ UrbanSound8K 10-fold 구조 ]
    fold 1~8: train (80%)
    fold 9:   val   (10%) — 하이퍼파라미터 튜닝 및 Early Stopping 기준
    fold 10:  test  (10%) — 최종 성능 평가 (학습 중 절대 사용 안 함)

    이 분할 방식은 원본 파일 단위로 분리되어 있어 데이터 누수 없음.
    """
    reserved = set(val_folds) | set(test_folds)

    test_df  = df[df["fold"].isin(test_folds)].copy()
    val_df   = df[df["fold"].isin(val_folds)].copy()
    train_df = df[~df["fold"].isin(reserved)].copy()

    logger.info(
        "Split — train: %d  val: %d  test: %d",
        len(train_df), len(val_df), len(test_df),
    )
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = split_df["class"].value_counts().to_dict()
        logger.info("  %s class dist: %s", split_name, dist)

    return train_df, val_df, test_df


def _df_to_arrays(df: pd.DataFrame, label_map: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """DataFrame에서 numpy 배열 X, y를 추출합니다.

    X: 임베딩 벡터들을 세로로 쌓아 (N, 1024) 배열 생성
    y: class_id를 dense label로 변환한 (N,) 정수 배열
    """
    X = np.stack(df["embedding"].values).astype(np.float32)
    y = df["class_id"].map(label_map).values.astype(np.int32)
    return X, y


# ---------------------------------------------------------------------------
# 클래스 가중치 계산
# ---------------------------------------------------------------------------

def get_class_weights(
    y_train: np.ndarray,
    label_map: dict[int, int],
    positive_boost: float = 1.0,
) -> dict[int, float]:
    """클래스 불균형을 보정하는 가중치를 계산합니다.

    [ 클래스 불균형 문제 ]
    UrbanSound8K에서 background 클립이 압도적으로 많아 모델이 background 편향될 수 있음.
    balanced 가중치: 각 클래스의 가중치 = 전체샘플수 / (클래스수 × 해당클래스샘플수)

    [ positive_boost ]
    안전에 직결된 양성 클래스(경적/사이렌)에 추가 가중치를 부여하여
    recall(재현율)을 높이는 전략. background는 가중치 변경 없음.

    예시 (boost=2.5):
        car_horn 1.37 → 3.42,  siren 0.63 → 1.58,  background 1.47 (유지)

    Args:
        y_train: 학습 데이터의 dense label 배열
        label_map: {원본 class_id → dense label} 매핑
        positive_boost: 양성 클래스 가중치 배수 (기본 1.0 = 변경 없음)

    Returns:
        Keras class_weight 인자에 넘길 {dense_label: weight} 딕셔너리
    """
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_dict = dict(zip(classes.tolist(), weights.tolist()))

    if positive_boost != 1.0:
        # background dense label은 label_map에서 가장 큰 인덱스 (build_label_map 보장)
        background_dense = max(label_map.values())
        for label, w in weight_dict.items():
            if label != background_dense:
                # 양성 클래스(경적/사이렌)에만 boost 적용
                weight_dict[label] = w * positive_boost

    return weight_dict


# ---------------------------------------------------------------------------
# 모델 구성
# ---------------------------------------------------------------------------

def build_model(cfg: dict, num_classes: int) -> Any:
    """YAMNet 임베딩 위에 얹히는 맞춤형 DNN 분류기를 생성합니다.

    [ 설계 원리 ]
    YAMNet이 1024차원 임베딩을 추출하면, 이 함수의 모델이 그 임베딩을
    3개 클래스로 분류합니다. YAMNet 자체는 학습하지 않고 고정(frozen)합니다.

    [ 아키텍처 (config.yaml의 classifier 섹션으로 조정 가능) ]
    Input(1024) → Dense + BatchNorm? + Dropout → ... → Dense(num_classes, softmax)

    [ Focal Loss ]
    감마(γ)>0으로 설정 시 쉬운 샘플(배경)의 loss를 크게 줄여
    경계 샘플(애매한 사이렌 소리)에 집중 학습.
    γ=0: 일반 CrossEntropy, γ=2: 어려운 샘플 4배 강조

    Args:
        cfg: config.yaml에서 파싱된 전체 설정 딕셔너리
        num_classes: 클래스 수 (데이터에서 자동 결정, config 수동 입력 불필요)

    Returns:
        컴파일된 tf.keras.Model
    """
    clf_cfg = cfg["classifier"]
    train_cfg = cfg["training"]

    hidden_units: list[int] = clf_cfg["hidden_units"]       # 예: [512, 256, 64]
    dropout_rate: float = clf_cfg["dropout_rate"]           # 예: 0.4
    use_batch_norm: bool = clf_cfg.get("batch_norm", True)  # 첫 레이어에 BatchNorm
    activation: str = clf_cfg.get("activation", "relu")     # 활성화 함수
    l2_reg: float = clf_cfg.get("l2_reg", 1e-4)             # L2 정규화 (과적합 방지)
    lr: float = train_cfg["learning_rate"]                   # 학습률

    reg = tf.keras.regularizers.l2(l2_reg)

    # ── 모델 구성 ──
    inputs = tf.keras.Input(shape=(1024,), name="yamnet_embedding")
    x = inputs

    for i, units in enumerate(hidden_units):
        # Dense → BatchNorm(선택) → 활성화 → Dropout 순서
        x = tf.keras.layers.Dense(
            units, activation=None, kernel_regularizer=reg, name=f"dense_{i}",
        )(x)
        if use_batch_norm and i == 0:
            # 첫 번째 레이어에만 BatchNorm: 학습 안정화 + 내부 공변량 이동 방지
            x = tf.keras.layers.BatchNormalization(name=f"bn_{i}")(x)
        x = tf.keras.layers.Activation(activation, name=f"act_{i}")(x)
        x = tf.keras.layers.Dropout(dropout_rate, name=f"drop_{i}")(x)

    output_activation: str = clf_cfg.get("output_activation", "softmax")
    use_sigmoid: bool = output_activation == "sigmoid"

    # 출력층: softmax(다중 클래스) 또는 sigmoid(각 클래스 독립)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation=output_activation,
        name="output",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="adas_classifier")

    if use_sigmoid:
        # sigmoid: 각 클래스를 독립적인 binary 분류 → one-hot 레이블 필요
        loss_fn = tf.keras.losses.BinaryCrossentropy()
        metrics: list = [tf.keras.metrics.BinaryAccuracy(name="accuracy")]
    else:
        # Focal Loss: γ=0 → 일반 CrossEntropy, γ>0 → 어려운 샘플에 집중
        focal_gamma: float = train_cfg.get("focal_loss_gamma", 0.0)
        if focal_gamma > 0:
            try:
                # TF 2.11+ 에 내장된 Focal Loss 사용
                loss_fn = tf.keras.losses.SparseCategoricalFocalCrossentropy(gamma=focal_gamma)
            except AttributeError:
                # 구버전 TF: 수동으로 Focal Loss 구현
                def focal_loss_fn(y_true: Any, y_pred: Any) -> Any:
                    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
                    # p_t: 정답 클래스의 예측 확률
                    p_t = tf.reduce_max(y_pred * tf.one_hot(
                        tf.cast(y_true, tf.int32), tf.shape(y_pred)[-1]
                    ), axis=-1)
                    # (1-p_t)^γ: 쉬운 샘플(p_t 높음)은 loss 대폭 감소
                    return tf.reduce_mean((1 - p_t) ** focal_gamma * ce)
                loss_fn = focal_loss_fn
        else:
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = ["accuracy"]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss_fn,
        metrics=metrics,
    )
    return model


# ---------------------------------------------------------------------------
# 콜백 설정
# ---------------------------------------------------------------------------

def build_callbacks(cfg: dict, checkpoint_path: Path) -> list:
    """Keras 학습 콜백을 config 기반으로 구성합니다.

    [ 콜백 목적 ]
    1. EarlyStopping: val_loss가 개선되지 않으면 조기 종료
       - patience 에폭 동안 개선 없으면 학습 중단 + 최적 가중치 복원
    2. ReduceLROnPlateau: val_loss 정체 시 학습률 자동 감소
       - 학습 곡선이 평탄해질 때 더 세밀한 조정
    3. ModelCheckpoint: val_loss 기준 최적 모델 자동 저장
       - 학습 도중 최선의 가중치를 파일로 보존
    """
    train_cfg = cfg["training"]
    es_cfg = train_cfg["early_stopping"]
    lr_cfg = train_cfg["reduce_lr"]

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=es_cfg["monitor"],                            # 'val_loss'
            patience=es_cfg["patience"],                          # 개선 없는 에폭 수
            restore_best_weights=es_cfg.get("restore_best_weights", True),
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=lr_cfg["monitor"],   # 'val_loss'
            factor=lr_cfg["factor"],     # 학습률 감소 비율 (예: 0.5 → 절반으로)
            patience=lr_cfg["patience"], # 감소 전 대기 에폭 수
            min_lr=lr_cfg["min_lr"],     # 최소 학습률 (이 이하로는 감소 안 함)
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,   # 최고 성능 시에만 저장
            verbose=1,
        ),
    ]


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------

def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_map: dict[int, int],
    class_map: dict[int, str],
    logger: logging.Logger,
    use_sigmoid: bool = False,
) -> dict[str, Any]:
    """테스트 세트에서 모델 성능을 평가하고 클래스별 지표를 출력합니다.

    [ 평가 지표 설명 ]
    - Recall(재현율): 실제 양성 중 맞게 탐지한 비율 → 놓치면 안 되는 사이렌/경적에 중요
    - Precision(정밀도): 탐지 양성 중 실제 양성 비율 → 오경보 비율
    - F1-score: Recall과 Precision의 조화평균
    - Confusion Matrix: 클래스별 혼동 상황 2D 시각화

    Args:
        model: 학습된 Keras 모델
        X_test: 테스트 임베딩 (N, 1024)
        y_test: 테스트 dense label (N,)
        label_map: {class_id: dense_label}
        class_map: {class_id: class_name} (사람이 읽기 쉬운 이름)
        logger: 로깅 인스턴스
        use_sigmoid: True이면 y_test를 one-hot으로 변환 후 evaluate()

    Returns:
        confusion_matrix, report, test_loss, test_accuracy 포함 딕셔너리
    """
    num_classes = len(label_map)
    # sigmoid 출력 모드에서는 레이블을 one-hot으로 변환해야 evaluate() 호환
    y_test_fit = (
        tf.keras.utils.to_categorical(y_test, num_classes) if use_sigmoid else y_test
    )
    loss, acc = model.evaluate(X_test, y_test_fit, verbose=0)
    logger.info("Test loss: %.4f  |  Test accuracy: %.4f", loss, acc)

    # argmax로 예측 클래스 결정
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    # dense label → 클래스 이름 역매핑
    inv_map = {v: k for k, v in label_map.items()}
    dense_labels = sorted(inv_map.keys())
    target_names = [class_map.get(inv_map[l], str(inv_map[l])) for l in dense_labels]

    cm = confusion_matrix(y_test, y_pred, labels=dense_labels)
    report = classification_report(
        y_test, y_pred, labels=dense_labels,
        target_names=target_names, output_dict=True,
    )

    logger.info("Confusion matrix:\n%s", cm)
    logger.info(
        "Classification report:\n%s",
        classification_report(y_test, y_pred, labels=dense_labels, target_names=target_names),
    )

    return {
        "confusion_matrix": cm,
        "report": report,
        "test_loss": float(loss),
        "test_accuracy": float(acc),
    }


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------

def save_training_plots(
    history: Any,
    cm: np.ndarray,
    label_map: dict[int, int],
    class_map: dict[int, str],
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    """학습 곡선(loss/accuracy)과 혼동 행렬 히트맵을 PNG로 저장합니다.

    [ 출력 파일 ]
    training_curves.png: 에폭별 train/val loss·accuracy 그래프
    confusion_matrix.png: 테스트 세트 예측 결과 혼동 행렬
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # GUI 없는 환경에서 렌더링 (서버/Raspberry Pi)
        import matplotlib.pyplot as plt  # type: ignore[import]
        import seaborn as sns  # type: ignore[import]
    except ImportError:
        logger.warning("matplotlib/seaborn not installed — skipping plots.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    hist = history.history

    # 학습/검증 loss·accuracy 그래프
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist["loss"], label="train")
    axes[0].plot(hist["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(hist["accuracy"], label="train")
    axes[1].plot(hist["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    curve_path = output_dir / "training_curves.png"
    fig.savefig(str(curve_path), dpi=120)
    plt.close(fig)
    logger.info("Training curves → %s", curve_path)

    # 혼동 행렬 히트맵
    inv_map = {v: k for k, v in label_map.items()}
    tick_labels = [class_map.get(inv_map[i], str(inv_map[i])) for i in sorted(inv_map)]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=tick_labels, yticklabels=tick_labels,
                cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Test Set)")
    fig.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    fig.savefig(str(cm_path), dpi=120)
    plt.close(fig)
    logger.info("Confusion matrix → %s", cm_path)


# ---------------------------------------------------------------------------
# 메인 학습 파이프라인
# ---------------------------------------------------------------------------

def run_classifier(config_path: str) -> None:
    """분류기 학습 파이프라인 전체를 실행합니다 (Step 4).

    [ 전체 흐름 ]
    1. embeddings.npz + manifest.csv 로드
    2. 데이터 누수 검사 (원본 파일이 여러 폴드에 걸치지 않는지)
    3. 레이블 맵 자동 결정 ({1→0, 8→1, 99→2})
    4. train/val/test 폴드 분할
    5. 클래스 가중치 계산 (불균형 보정)
    6. DNN 모델 생성 및 컴파일
    7. 학습 (EarlyStopping + ModelCheckpoint)
    8. 테스트 세트 평가
    9. 학습 곡선 + 혼동 행렬 저장
    10. 최종 성능 요약 출력

    Args:
        config_path: config.yaml 경로
    """
    cfg = load_config(config_path)
    logger = setup_logging(cfg, __name__)

    project_root = Path(config_path).parent
    ds_cfg = cfg["dataset"]
    train_cfg = cfg["training"]
    export_cfg = cfg["export"]
    yamnet_cfg = cfg["yamnet"]

    # 재현성을 위한 난수 시드 고정
    seed: int = train_cfg["seed"]
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # 경로 설정
    processed_dir = project_root / ds_cfg["processed_dir"]
    npz_path = project_root / yamnet_cfg["cache_dir"] / "embeddings.npz"
    manifest_path = processed_dir / "manifest.csv"
    model_dir = project_root / export_cfg["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = model_dir / export_cfg["keras_filename"]  # 최적 모델 저장 경로
    plot_dir = project_root / "notebooks" / "plots"

    val_folds: list[int] = ds_cfg["val_folds"]    # [9]
    test_folds: list[int] = ds_cfg["test_folds"]  # [10]

    # config.yaml에서 클래스 이름 자동 로드 ({class_id: class_name})
    class_map = get_class_map(cfg)

    # ── 1. 데이터셋 로드 ──
    df = load_embedding_dataset(npz_path, manifest_path, logger)

    # ── 2. 데이터 누수 검사 ──
    verify_no_leakage(df, logger)

    # ── 3. 레이블 매핑 — 데이터에서 자동 결정 ──
    label_map = build_label_map(df)
    num_classes = len(label_map)
    logger.info(
        "Label map (class_id → dense): %s  |  num_classes=%d (auto-detected)",
        label_map, num_classes,
    )
    for cid, dense in label_map.items():
        name = class_map.get(cid, f"id_{cid}")
        logger.info("  class %d → dense %d  (%s)", cid, dense, name)

    # ── 4. 폴드 분할 ──
    train_df, val_df, test_df = fold_split(df, val_folds, test_folds, logger)

    X_train, y_train = _df_to_arrays(train_df, label_map)
    X_val,   y_val   = _df_to_arrays(val_df,   label_map)
    X_test,  y_test  = _df_to_arrays(test_df,  label_map)

    # ── 5. 클래스 가중치 계산 ──
    positive_boost: float = train_cfg.get("class_weight_positive_boost", 1.0)
    class_weights = get_class_weights(y_train, label_map, positive_boost)
    logger.info(
        "Class weights (positive_boost=%.1f): %s", positive_boost, class_weights
    )

    # ── 6. 모델 생성 (num_classes 자동 주입) ──
    use_sigmoid: bool = cfg["classifier"].get("output_activation", "softmax") == "sigmoid"
    model = build_model(cfg, num_classes=num_classes)
    model.summary(print_fn=logger.info)

    # ── 7. 학습 ──
    callbacks = build_callbacks(cfg, checkpoint_path)

    if use_sigmoid:
        # sigmoid 출력: one-hot 레이블 + sample_weight 방식
        logger.info("sigmoid 출력 모드: one-hot 레이블 + sample_weight 사용")
        y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes)
        y_val_oh   = tf.keras.utils.to_categorical(y_val,   num_classes)
        # class_weight 대신 각 샘플에 직접 가중치 부여
        sample_weight_train = np.array([class_weights[int(y)] for y in y_train], dtype=np.float32)
        history = model.fit(
            X_train, y_train_oh,
            validation_data=(X_val, y_val_oh),
            epochs=train_cfg["epochs"],
            batch_size=train_cfg["batch_size"],
            sample_weight=sample_weight_train,
            callbacks=callbacks,
            verbose=2,
        )
    else:
        # softmax 출력: sparse label + class_weight 방식 (일반적)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=train_cfg["epochs"],
            batch_size=train_cfg["batch_size"],
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=2,
        )
    logger.info("Training complete — best model saved to %s", checkpoint_path)

    # ── 8. 테스트 세트 평가 ──
    results = evaluate_model(model, X_test, y_test, label_map, class_map, logger,
                             use_sigmoid=use_sigmoid)

    # ── 9. 그래프 저장 ──
    save_training_plots(history, results["confusion_matrix"], label_map, class_map, plot_dir, logger)

    # ── 10. 최종 요약 ──
    logger.info("=" * 60)
    logger.info("CLASSIFIER TRAINING COMPLETE")
    logger.info("  Num classes   : %d (%s)", num_classes, list(class_map.values()))
    logger.info("  Test accuracy : %.4f", results["test_accuracy"])
    logger.info("  Test loss     : %.4f", results["test_loss"])
    for cls_label, metrics in results["report"].items():
        if isinstance(metrics, dict):
            logger.info(
                "  %-15s  P=%.3f  R=%.3f  F1=%.3f  N=%d",
                cls_label,
                metrics.get("precision", 0),
                metrics.get("recall", 0),
                metrics.get("f1-score", 0),
                int(metrics.get("support", 0)),
            )
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI 진입점
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ADAS Sound Detector – Classifier Training")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    run_classifier(_parse_args().config)
