"""[심화2] AI 분리수거 도우미 - CLIP 제로샷 분류 모듈.

CLIP은 이미지와 문장을 같은 512차원 공간의 벡터로 바꾸는 모델입니다.
"a photo of a plastic bottle" 같은 설명 문장들과 사진의 벡터를 비교해서
가장 비슷한 설명을 고르면, 따로 학습하지 않고도(제로샷) 분류할 수 있습니다.

사용 모델: Xenova/clip-vit-base-patch32 (OpenAI CLIP ViT-B/32의 ONNX 변환본, MIT 라이선스)
  - vision_model_quantized.onnx : 사진 -> 512차원 벡터 (INT8 양자화, 89MB)
  - text_model_quantized.onnx   : 문장 -> 512차원 벡터 (INT8 양자화, 64MB)
  - tokenizer.json              : 문장 -> 토큰 ID

모델 다운로드(download_models)는 채팅용 한국어 문장 임베딩 모델(recycle_chat.py에서 사용)도 함께 받습니다.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from tokenizers import Tokenizer

CLIP_BASE = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main"
# 한국어 질문 이해용 다국어 문장 임베딩 모델 (원본: sentence-transformers, Apache-2.0)
KO_BASE = "https://huggingface.co/Xenova/paraphrase-multilingual-MiniLM-L12-v2/resolve/main"
MODEL_DIR = Path(__file__).resolve().parent / "models2"
FILES = {
    "vision": (f"{CLIP_BASE}/onnx/vision_model_quantized.onnx", "vision_model_quantized.onnx", 80_000_000),
    "text": (f"{CLIP_BASE}/onnx/text_model_quantized.onnx", "text_model_quantized.onnx", 60_000_000),
    "tokenizer": (f"{CLIP_BASE}/tokenizer.json", "tokenizer.json", 1_000_000),
    "ko_text": (f"{KO_BASE}/onnx/model_quantized.onnx", "text_ko_quantized.onnx", 100_000_000),
    "ko_tokenizer": (f"{KO_BASE}/tokenizer.json", "tokenizer_ko.json", 10_000_000),
}  # 이름: (다운로드 URL, 저장할 파일 이름, 정상 파일의 최소 크기)

# CLIP 학습 때 쓴 이미지 정규화 값 (preprocessor_config.json)
IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
IMAGE_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
IMAGE_SIZE = 224
LOGIT_SCALE = 100.0  # CLIP이 학습한 온도 값: 유사도 차이를 확률 차이로 키워 줌


@dataclass(frozen=True)
class Category:
    """분리수거 분류 하나. prompts는 (CLIP에 줄 영어 설명, 화면에 보일 한국어 물건 이름)."""

    name: str
    icon: str
    bin: str  # 버리는 곳
    how: list[str]  # 버리는 방법
    prompts: list[tuple[str, str]]


# 일반적인 국내 분리배출 기준 (지자체마다 세부 기준이 다를 수 있음)
CATEGORIES: list[Category] = [
    Category(
        "플라스틱", ":material/water_bottle:", "플라스틱 분리수거함",
        ["내용물을 비우고 물로 헹궈요.", "라벨과 뚜껑은 떼서 따로 배출해요.",
         "투명 페트병은 찌그러뜨려 투명 페트병 전용 수거함에 넣어요."],
        [("a plastic bottle", "페트병"), ("a plastic container", "플라스틱 용기"),
         ("a plastic cup", "플라스틱 컵"), ("a plastic shampoo bottle", "샴푸통")],
    ),
    Category(
        "캔·고철", ":material/takeout_dining:", "캔·고철 분리수거함",
        ["내용물을 비우고 헹군 뒤 가능하면 납작하게 눌러요.",
         "부탄가스·스프레이 캔은 구멍을 뚫어 가스를 완전히 뺀 뒤 배출해요."],
        [("an aluminum soda can", "음료 캔"), ("a tin can", "통조림 캔"),
         ("a metal pot", "냄비"), ("a spray can", "스프레이 캔")],
    ),
    Category(
        "유리병", ":material/liquor:", "유리병 분리수거함",
        ["뚜껑을 분리하고 내용물을 비워 헹궈요.",
         "소주·맥주병은 빈용기 보증금 대상이라 마트에 반납하면 돈을 돌려받아요.",
         "깨진 유리, 거울, 도자기는 유리병이 아니에요 (신문지에 싸서 일반쓰레기)."],
        [("a glass bottle", "유리병"), ("a glass jar", "유리 병조림 병"),
         ("a beer bottle", "맥주병"), ("a wine bottle", "와인병")],
    ),
    Category(
        "종이", ":material/description:", "종이 분리수거함",
        ["테이프, 스테이플러 심, 비닐 코팅을 제거해요.", "상자는 펴서 차곡차곡 묶어서 배출해요.",
         "영수증, 코팅된 종이, 젖거나 기름 묻은 종이는 일반쓰레기예요."],
        [("a cardboard box", "골판지 상자"), ("a newspaper", "신문지"),
         ("a paper bag", "종이봉투"), ("a stack of paper documents", "종이 문서"),
         ("a book", "책")],
    ),
    Category(
        "종이팩", ":material/local_drink:", "종이팩 전용 수거함",
        ["내용물을 비우고 헹군 뒤 펼쳐서 말려요.",
         "일반 종이와 섞지 말고 종이팩끼리 모아요 (주민센터에서 휴지로 교환해 주기도 해요)."],
        [("a milk carton", "우유팩"), ("a juice carton", "주스팩")],
    ),
    Category(
        "비닐", ":material/shopping_bag:", "비닐 분리수거함",
        ["음식물 등 이물질을 털어내거나 헹궈요.", "흩날리지 않게 여러 장을 한 봉투에 모아서 배출해요."],
        [("a plastic bag", "비닐봉지"), ("a snack wrapper", "과자 봉지"),
         ("plastic wrap film", "랩·비닐 포장재"), ("a ramen noodle package", "라면 봉지")],
    ),
    Category(
        "스티로폼", ":material/inventory_2:", "스티로폼 분리수거함",
        ["테이프, 스티커, 운송장을 떼요.", "음식물이 묻었거나 색이 있는 스티로폼은 일반쓰레기예요."],
        [("white styrofoam packaging", "스티로폼 완충재"), ("a styrofoam box", "스티로폼 상자"),
         ("a styrofoam food tray", "스티로폼 접시")],
    ),
    Category(
        "음식물", ":material/compost:", "음식물 쓰레기통",
        ["물기를 최대한 빼서 버려요.",
         "닭뼈, 생선뼈, 달걀·조개 껍데기, 복숭아 씨, 티백은 일반쓰레기예요."],
        [("food waste", "음식물 쓰레기"), ("leftover food on a plate", "남은 음식"),
         ("a banana peel", "바나나 껍질"), ("fruit and vegetable scraps", "과일·채소 찌꺼기")],
    ),
    Category(
        "폐건전지·전자", ":material/battery_full:", "폐건전지·소형가전 전용 수거함",
        ["폐건전지는 주민센터·아파트의 전용 수거함에 넣어요.",
         "휴대폰, 충전기 등 소형 가전은 폐가전 무상 수거나 전용 수거함을 이용해요."],
        [("a battery", "건전지"), ("a mobile phone", "휴대폰"),
         ("a phone charger cable", "충전기")],
    ),
    Category(
        "일반쓰레기", ":material/delete:", "종량제 봉투",
        ["재활용이 안 되는 쓰레기는 종량제 봉투에 담아 버려요.",
         "깨진 유리나 날카로운 물건은 신문지에 싸서 다치지 않게 버려요."],
        [("a used tissue", "사용한 휴지"), ("a disposable face mask", "일회용 마스크"),
         ("a toothbrush", "칫솔"), ("a broken ceramic cup", "깨진 도자기"),
         ("a cigarette butt", "담배꽁초")],
    ),
]


# ---------------------------------------------------------------------------
# 모델 다운로드 / 로딩
# ---------------------------------------------------------------------------
def download_models(model_dir: Path = MODEL_DIR) -> None:
    """모델 파일이 없으면 Hugging Face에서 내려받습니다."""
    model_dir.mkdir(exist_ok=True)
    for url, filename, min_size in FILES.values():
        path = model_dir / filename
        if path.exists() and path.stat().st_size >= min_size:
            continue  # 이미 받아 둠
        tmp = path.with_suffix(".download")
        urllib.request.urlretrieve(url, tmp)
        if tmp.stat().st_size < min_size:
            tmp.unlink()
            raise RuntimeError(f"다운로드한 파일이 올바르지 않습니다: {url}")
        tmp.replace(path)


def model_path(key: str) -> Path:
    """FILES에 등록된 모델 파일의 로컬 경로."""
    return MODEL_DIR / FILES[key][1]


class RecycleClassifier:
    """CLIP 이미지/텍스트 모델을 묶어서 분리수거 분류를 수행합니다."""

    def __init__(self) -> None:
        download_models()
        opts = ["CPUExecutionProvider"]
        self.vision = ort.InferenceSession(str(model_path("vision")), providers=opts)
        self.text = ort.InferenceSession(str(model_path("text")), providers=opts)
        self.tokenizer = Tokenizer.from_file(str(model_path("tokenizer")))

        # 모든 설명 문장의 벡터를 미리 한 번만 계산해 둡니다 (사진마다 다시 계산할 필요 없음).
        self.prompt_items: list[tuple[int, str]] = []  # (분류 번호, 한국어 물건 이름)
        sentences = []
        for ci, cat in enumerate(CATEGORIES):
            for english, korean in cat.prompts:
                self.prompt_items.append((ci, korean))
                sentences.append(f"a photo of {english}.")
        self.prompt_embeds = self._embed_texts(sentences)  # (설명 수, 512)

    # --- 텍스트 -> 벡터 ------------------------------------------------------
    def _embed_texts(self, sentences: list[str]) -> np.ndarray:
        embeds = []
        for s in sentences:  # 길이가 서로 달라서 한 문장씩 처리 (시작할 때 한 번만 실행)
            ids = np.array([self.tokenizer.encode(s).ids], dtype=np.int64)
            embeds.append(self.text.run(["text_embeds"], {"input_ids": ids})[0][0])
        return _normalize(np.stack(embeds))

    # --- 사진 전처리 ---------------------------------------------------------
    @staticmethod
    def preprocess(image: Image.Image) -> np.ndarray:
        """사진을 CLIP 입력 형식 (1, 3, 224, 224)로 바꿉니다.

        1) 휴대폰 사진의 회전 정보(EXIF) 반영, RGB 변환
        2) 짧은 변을 224로 맞춰 축소 (비율 유지, BICUBIC)
        3) 가운데 224x224 잘라내기
        4) 0~1로 바꾼 뒤 CLIP 학습 때의 평균/표준편차로 정규화
        5) (높이, 너비, 채널) -> (1, 채널, 높이, 너비)
        """
        image = ImageOps.exif_transpose(image).convert("RGB")
        w, h = image.size
        scale = IMAGE_SIZE / min(w, h)
        image = image.resize((max(IMAGE_SIZE, round(w * scale)), max(IMAGE_SIZE, round(h * scale))),
                             Image.Resampling.BICUBIC)
        w, h = image.size
        left, top = (w - IMAGE_SIZE) // 2, (h - IMAGE_SIZE) // 2
        image = image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))
        arr = (np.asarray(image, dtype=np.float32) / 255.0 - IMAGE_MEAN) / IMAGE_STD
        return arr.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    # --- 분류 ----------------------------------------------------------------
    def classify(self, image: Image.Image) -> dict:
        """사진을 분류하고 결과를 반환합니다.

        반환 dict
          probs      : 분류별 확률 (len(CATEGORIES),)
          best       : 가장 확률이 높은 분류 번호
          item       : 가장 비슷한 물건 이름 (예: "페트병")
          input_image: 모델에 실제로 들어간 224x224 이미지 (화면 표시용)
        """
        pixel_values = self.preprocess(image)
        image_embed = _normalize(self.vision.run(["image_embeds"], {"pixel_values": pixel_values})[0])

        # 사진과 각 설명 문장의 코사인 유사도
        sims = (image_embed @ self.prompt_embeds.T)[0]

        # 분류마다 "가장 비슷한 설명 하나"의 유사도를 그 분류의 점수로 사용합니다.
        # (확률을 더하는 방식은 설명 문장이 많은 분류가 유리해지는 문제가 있어서 max 사용)
        cat_ids = np.array([ci for ci, _ in self.prompt_items])
        cat_sims = np.array([sims[cat_ids == ci].max() for ci in range(len(CATEGORIES))])
        probs = _softmax(LOGIT_SCALE * cat_sims)  # 분류별 확률 (합 = 1)

        best = int(probs.argmax())
        in_best = np.where(cat_ids == best)[0]
        best_prompt = int(in_best[sims[in_best].argmax()])  # 그 분류 안에서 가장 비슷한 물건
        return {
            "probs": probs,
            "best": best,
            "item": self.prompt_items[best_prompt][1],
            "input_image": _to_display(pixel_values),
        }


# ---------------------------------------------------------------------------
# 보조 함수
# ---------------------------------------------------------------------------
def _normalize(x: np.ndarray) -> np.ndarray:
    """벡터 길이를 1로 맞춥니다 (내적 = 코사인 유사도가 되도록)."""
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _to_display(pixel_values: np.ndarray) -> Image.Image:
    """정규화된 입력 텐서를 다시 눈으로 볼 수 있는 이미지로 되돌립니다."""
    arr = pixel_values[0].transpose(1, 2, 0) * IMAGE_STD + IMAGE_MEAN
    return Image.fromarray((arr.clip(0, 1) * 255).astype(np.uint8))
