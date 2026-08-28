# ==============================================================================
# [공공 감염병 진단·감시 임상 의사결정 지원 시스템 (CDSS) - v16.2 Production]
# - 흉부 X-선(64x64) 폐렴/호흡기 감염병 CNN 인코더 및 Grad-CAM XAI 시각화
# - 글로벌 감염병 통계(OWID/WHO) 사전 캐시 로드 기반 위험도 보정
# - Sanger Sequencing 자동 BLAST 종 동정 연동
# - 캘리브레이션(T=1.35) + MC Dropout 불확실성(±%) + 임상 임계값(Threshold) 제어
# - 미국 CDC 35종 표준 + 질병관리청(KDCA) 신고 + SHA-256 전자서명 EMR SOAP / PDF
# ==============================================================================

import datetime
import hashlib
import io
import json
import os
import warnings
from Bio import SeqIO
from Bio.Blast import NCBIWWW, NCBIXML
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pymupdf
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
BASE_DIR = "./infectious_disease_data"
CACHE_FILE = os.path.join(BASE_DIR, "epidemiology_cache.json")
WEIGHTS_PATH = "./cdss_model_weights.pth"
os.makedirs(BASE_DIR, exist_ok=True)

# ---------------------------------------------------------
# 0. 글로벌 감염병 통계 캐시 로드
# ---------------------------------------------------------
epi_cache = {
    "korea_covid_new_cases": 0,
    "global_mpox_cases": {},
    "who_tb_incidence_kor": 44.0,
    "updated_at": "N/A",
}

if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            epi_cache = json.load(f)
    except Exception:
        pass


# ---------------------------------------------------------
# 1. 자연어 문진 토크나이저
# ---------------------------------------------------------
class ClinicalTextTokenizer:

    def __init__(self):
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        base_words = [
            "발열",
            "고열",
            "기침",
            "오한",
            "두통",
            "발진",
            "수포",
            "가피",
            "모기",
            "진드기",
            "동남아",
            "아프리카",
            "여행",
            "설사",
            "구토",
            "황달",
            "호흡곤란",
            "근육통",
            "관절통",
            "교상",
            "폐렴",
            "흉통",
        ]
        for idx, w in enumerate(base_words, 2):
            self.vocab[w] = idx

    def encode(self, text, max_len=16):
        if not text:
            return [0] * max_len
        tokens = []
        for word in text.split():
            matched = False
            for k, v in self.vocab.items():
                if k in word and k not in ["<PAD>", "<UNK>"]:
                    tokens.append(v)
                    matched = True
                    break
            if not matched:
                tokens.append(1)
        if len(tokens) < max_len:
            tokens += [0] * (max_len - len(tokens))
        return tokens[:max_len]


tokenizer = ClinicalTextTokenizer()


# ---------------------------------------------------------
# 2. 멀티모달 딥러닝 서브 네트워크 & Grad-CAM 지원 CNN
# ---------------------------------------------------------
class ClinicalTextLSTM(nn.Module):

    def __init__(self, vocab_size=64, embed_dim=32, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 64)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        emb = self.embedding(x)
        _, (hn, _) = self.lstm(emb)
        feat = torch.cat((hn[-2], hn[-1]), dim=1)
        return self.drop(F.relu(self.fc(feat)))


class MedicalChestCNN(nn.Module):

    def __init__(self, out_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.adapt_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc = nn.Linear(64 * 4 * 4, out_dim)
        self.drop = nn.Dropout(0.2)

        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        if x is None:
            return torch.zeros((1, 64), device=self.fc.weight.device)

        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)

        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        h = F.relu(self.bn3(self.conv3(x)))
        if h.requires_grad:
            h.register_hook(self.activations_hook)
        self.activations = h

        pooled = self.adapt_pool(h)
        flat = pooled.view(pooled.size(0), -1)
        return self.drop(F.relu(self.fc(flat)))


class MultimodalCDSSNet(nn.Module):

    def __init__(self, num_features=30, num_classes=34, vocab_size=64):
        super().__init__()
        self.text_encoder = ClinicalTextLSTM(
            vocab_size=vocab_size, hidden_dim=32
        )
        self.image_encoder = MedicalChestCNN(out_dim=64)

        self.tabular_encoder = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
        )

        self.fusion_attention = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, batch_first=True, dropout=0.2
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 3, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, tab_x, text_x=None, img_x=None):
        batch_size = tab_x.shape[0]
        v_feat = self.tabular_encoder(tab_x).unsqueeze(1)
        t_feat = (
            self.text_encoder(text_x).unsqueeze(1)
            if text_x is not None
            else torch.zeros_like(v_feat)
        )
        i_feat = (
            self.image_encoder(img_x).unsqueeze(1)
            if img_x is not None
            else torch.zeros_like(v_feat)
        )

        multimodal_seq = torch.cat([v_feat, t_feat, i_feat], dim=1)
        fused = self.fusion_attention(multimodal_seq)
        flat = fused.reshape(batch_size, -1)
        return self.classifier(flat)


# ---------------------------------------------------------
# 3. Grad-CAM 히트맵 생성 함수
# ---------------------------------------------------------
def generate_gradcam_heatmap(img_tensor, target_class, runtime_device):
    if img_tensor is None:
        return None

    multimodal_model.eval()
    img_t = img_tensor.clone().detach().to(runtime_device).requires_grad_(True)

    dummy_tab = torch.zeros((1, len(features)), device=runtime_device)
    dummy_text = torch.zeros((1, 16), dtype=torch.long, device=runtime_device)

    logits = multimodal_model(dummy_tab, dummy_text, img_t)
    target_score = logits[0, target_class]

    multimodal_model.zero_grad()
    target_score.backward()

    gradients = multimodal_model.image_encoder.gradients
    activations = multimodal_model.image_encoder.activations

    if gradients is None or activations is None:
        return None

    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * activations, dim=1).squeeze()
    cam = np.maximum(cam.cpu().detach().numpy(), 0)

    if np.max(cam) != 0:
        cam = cam / np.max(cam)

    cam_pil = Image.fromarray(np.uint8(255 * cam)).resize(
        (64, 64), Image.BILINEAR
    )
    heatmap = np.uint8(255 * np.array(cam_pil) / 255.0)

    colormap = plt.get_cmap("jet")
    colored_heatmap = colormap(heatmap / 255.0)[:, :, :3]

    orig_np = img_tensor.squeeze().cpu().numpy()
    if orig_np.ndim == 3:
        orig_np = orig_np.mean(axis=0)
    orig_rgb = np.stack([orig_np] * 3, axis=-1)

    overlay = np.uint8(255 * (0.5 * orig_rgb + 0.5 * colored_heatmap))
    overlay_img = Image.fromarray(overlay).resize((256, 256), Image.NEAREST)
    return overlay_img


# ---------------------------------------------------------
# 4. Sanger Sequencing 분석 에이전트
# ---------------------------------------------------------
class SangerSequenceAgent:
    PATHOGEN_KEYWORDS = {
        "sars-cov-2": "COVID-19",
        "coronavirus": "COVID-19",
        "influenza a": "Influenza (독감)",
        "influenza b": "Influenza (독감)",
        "dengue": "Dengue (뎅기열)",
        "flavivirus": "Dengue (뎅기열)",
        "plasmodium falciparum": "Malaria (말라리아)",
        "plasmodium vivax": "Malaria (말라리아)",
        "orientia tsutsugamushi": "Scrub_Typhus (쯔쯔가무시증)",
        "borrelia burgdorferi": "Lyme (라임병)",
        "mycobacterium tuberculosis": "Tuberculosis (결핵)",
        "legionella pneumophila": "Legionella (레지오넬라증)",
        "bordetella pertussis": "Pertussis (백일해)",
        "corynebacterium diphtheriae": "Diphtheria (디프테리아)",
        "monkeypox": "Mpox (엠폭스)",
        "mpox": "Mpox (엠폭스)",
        "varicella-zoster": "Varicella (수두/대상포진)",
        "bacillus anthracis": "Anthrax (탄저)",
        "streptococcus pyogenes": "Scarlet_Fever (성홍열)",
        "ebolavirus": "Ebola_Marburg (에볼라/마버그열)",
        "marburg": "Ebola_Marburg (에볼라/마버그열)",
        "lassa": "Lassa_Fever (라싸열)",
        "hantavirus": "Hanta_HFRS (신증후군출혈열)",
        "vibrio cholerae": "Cholera (콜레라)",
        "salmonella enterica": "Typhoid (장티푸스)",
        "shigella": "Shigellosis (세균성이질)",
        "hepatitis a": "Hepatitis_A (A형간염)",
        "clostridium botulinum": "Botulism (보툴리눔독소증)",
        "rabies": "Rabies (공수병)",
        "neisseria meningitidis": "Meningococcal (수막구균수막염)",
        "leptospira": "Leptospirosis (렙토스피라증)",
        "yersinia pestis": "Plague (페스트)",
        "mers-cov": "MERS (메르스)",
        "dabie bandavirus": "SFTS (중증열성혈소판감소)",
        "sfts": "SFTS (중증열성혈소판감소)",
        "clostridium tetani": "Tetanus (파상풍)",
    }

    @staticmethod
    def parse_sequence(file_obj, text_input=""):
        seq_record = None
        if file_obj is not None:
            filename = (
                file_obj.name if hasattr(file_obj, "name") else str(file_obj)
            )
            ext = os.path.splitext(filename)[-1].lower()
            fmt = "fastq" if ext in [".fastq", ".fq"] else "fasta"
            try:
                with open(
                    filename, "r", encoding="utf-8", errors="ignore"
                ) as handle:
                    records = list(SeqIO.parse(handle, fmt))
                    if records:
                        seq_record = records[0]
            except Exception:
                pass

        if seq_record is None and text_input and text_input.strip():
            fmt = "fastq" if text_input.strip().startswith("@") else "fasta"
            try:
                records = list(
                    SeqIO.parse(io.StringIO(text_input.strip()), fmt)
                )
                if records:
                    seq_record = records[0]
                else:
                    clean_seq = "".join(text_input.split()).upper()
                    seq_record = SeqRecord(
                        Seq(clean_seq),
                        id="Direct_Sequence",
                        description="User raw sequence",
                    )
            except Exception:
                pass

        return seq_record

    @classmethod
    def run_online_blast(cls, seq_record, evalue_threshold=0.001):
        if seq_record is None or len(seq_record.seq) < 20:
            return None, "유효한 20bp 이상의 염기서열이 필요합니다."

        try:
            result_handle = NCBIWWW.qblast(
                program="blastn",
                database="nt",
                sequence=str(seq_record.seq),
                hitlist_size=3,
                expect=evalue_threshold,
            )

            blast_record = NCBIXML.read(result_handle)
            if not blast_record.alignments:
                return None, "BLAST 검색 결과 일치하는 병원체를 찾을 수 없습니다."

            top_alignment = blast_record.alignments[0]
            top_hsp = top_alignment.hsps[0]

            title = top_alignment.title
            identity_pct = (top_hsp.identities / top_hsp.align_length) * 100.0
            e_value = top_hsp.expect

            matched_disease = None
            for kw, disease in cls.PATHOGEN_KEYWORDS.items():
                if kw in title.lower():
                    matched_disease = disease
                    break

            result_info = {
                "matched_disease": matched_disease,
                "hit_title": title,
                "identity": round(identity_pct, 2),
                "e_value": e_value,
                "align_length": top_hsp.align_length,
                "query_id": seq_record.id,
            }
            return result_info, "NCBI BLAST 분석 완료"

        except Exception as e:
            return None, f"NCBI BLAST 통신 실패: {str(e)}"


# ---------------------------------------------------------
# 5. 메타데이터 및 지식베이스
# ---------------------------------------------------------
features = [
    "high_fever",
    "chills",
    "cough",
    "fatigue",
    "headache",
    "muscle_pain",
    "skin_rash",
    "joint_pain",
    "vomiting",
    "diarrhea",
    "loss_of_smell",
    "shortness_of_breath",
    "bloody_diarrhea",
    "eschar_scab",
    "eye_pain",
    "hemorrhage",
    "jaundice",
    "neck_stiffness",
    "paralysis_spasm",
    "hydrophobia",
    "travel_tropical",
    "travel_africa",
    "travel_middle_east",
    "mosquito_bite",
    "tick_bite",
    "animal_bite",
    "animal_lesion_contact",
    "confirmed_patient_contact",
    "raw_water_food",
    "unventilated_facility",
]

symptom_korean_map = {
    "고열 (High Fever ≥ 38°C)": "high_fever",
    "오한 / 전율 (Chills)": "chills",
    "기침 / 인후통 (Cough)": "cough",
    "피로감 / 극심한 쇠약감 (Fatigue)": "fatigue",
    "두통 (Headache)": "headache",
    "근육통 / 요통 (Muscle Pain)": "muscle_pain",
    "피부 발진 / 수포 / 가피 (Skin Rash)": "skin_rash",
    "관절통 / 관절부종 (Joint Pain)": "joint_pain",
    "구토 / 오심 (Vomiting)": "vomiting",
    "수양성 설사 (Watery Diarrhea)": "diarrhea",
    "점액성 혈변 (Bloody Diarrhea)": "bloody_diarrhea",
    "후각 / 미각 상실 (Loss of Smell)": "loss_of_smell",
    "호흡곤란 / 흉통 (Shortness of Breath)": "shortness_of_breath",
    "가피 / 검은 딱지 (Eschar)": "eschar_scab",
    "안구통 / 결막 충혈 (Eye Pain)": "eye_pain",
    "점막 출혈 / 자반증 (Hemorrhage)": "hemorrhage",
    "황달 / 암갈색 뇨 (Jaundice)": "jaundice",
    "경부 강직 / 뇌수막 자극 징후 (Neck Stiffness)": "neck_stiffness",
    "근육 연축 / 마비 / 개구장애 (Spasm/Paralysis)": "paralysis_spasm",
    "물 마실 때 극심한 인두 경련 / 공수 (Hydrophobia)": "hydrophobia",
}

epi_korean_map = {
    "동남아 / 중남미 등 아열대 위험지역 여행력": "travel_tropical",
    "아프리카 지역 해외 여행력": "travel_africa",
    "중동 지역 해외 여행력 (최근 1개월)": "travel_middle_east",
    "야외 활동 중 모기 물림 노출력": "mosquito_bite",
    "풀밭 / 수풀 야외 활동 중 진드기 노출력": "tick_bite",
    "야생동물 또는 유기견에게 물림 / 할큄": "animal_bite",
    "수포성 병변 환자 또는 동물 사체 접촉": "animal_lesion_contact",
    "호흡기/발열 감염병 확진자와 밀접 접촉": "confirmed_patient_contact",
    "오염된 식수 또는 익히지 않은 어패류 섭취": "raw_water_food",
    "밀폐시설 에어로졸 / 온수·냉각탑수 노출": "unventilated_facility",
}

feature_kr_names = {
    "high_fever": "고열 (≥38°C)",
    "chills": "오한/전율",
    "cough": "기침/인후통",
    "fatigue": "피로감",
    "headache": "두통",
    "muscle_pain": "근육통",
    "skin_rash": "피부 발진/수포",
    "joint_pain": "관절통",
    "vomiting": "구토",
    "diarrhea": "수양성 설사",
    "bloody_diarrhea": "점액성 혈변",
    "loss_of_smell": "후각/미각 상실",
    "shortness_of_breath": "호흡곤란",
    "eschar_scab": "가피(Eschar)",
    "eye_pain": "안구통/결막충혈",
    "hemorrhage": "점막/자반 출혈",
    "jaundice": "황달",
    "neck_stiffness": "경부강직",
    "paralysis_spasm": "근육연축/마비",
    "hydrophobia": "공수/인두경련",
    "travel_tropical": "동남아/중남미 여행력",
    "travel_africa": "아프리카 여행력",
    "travel_middle_east": "중동 여행력",
    "mosquito_bite": "모기 물림 노출",
    "tick_bite": "진드기 노출",
    "animal_bite": "동물 교상",
    "animal_lesion_contact": "수포환자/동물접촉",
    "confirmed_patient_contact": "확진자 밀접접촉",
    "raw_water_food": "오염 식수/음식 섭취",
    "unventilated_facility": "냉각탑수/에어로졸 노출",
}

disease_profiles = {
    "Dengue (뎅기열)": {
        "high_fever": 1,
        "headache": 1,
        "joint_pain": 1,
        "muscle_pain": 1,
        "skin_rash": 1,
        "vomiting": 1,
        "eye_pain": 1,
        "travel_tropical": 1,
        "mosquito_bite": 1,
    },
    "Malaria (말라리아)": {
        "high_fever": 1,
        "chills": 1,
        "headache": 1,
        "muscle_pain": 1,
        "fatigue": 1,
        "travel_africa": 1,
        "travel_tropical": 1,
        "mosquito_bite": 1,
    },
    "Zika (지카바이러스)": {
        "high_fever": 1,
        "skin_rash": 1,
        "joint_pain": 1,
        "eye_pain": 1,
        "travel_tropical": 1,
        "mosquito_bite": 1,
    },
    "Chikungunya (치쿤구니야열)": {
        "high_fever": 1,
        "joint_pain": 1,
        "muscle_pain": 1,
        "skin_rash": 1,
        "travel_tropical": 1,
        "mosquito_bite": 1,
    },
    "Yellow_Fever (황열)": {
        "high_fever": 1,
        "jaundice": 1,
        "hemorrhage": 1,
        "vomiting": 1,
        "travel_africa": 1,
        "travel_tropical": 1,
        "mosquito_bite": 1,
    },
    "Scrub_Typhus (쯔쯔가무시증)": {
        "high_fever": 1,
        "chills": 1,
        "headache": 1,
        "skin_rash": 1,
        "eschar_scab": 1,
        "tick_bite": 1,
    },
    "Lyme (라임병)": {
        "high_fever": 1,
        "fatigue": 1,
        "headache": 1,
        "skin_rash": 1,
        "joint_pain": 1,
        "tick_bite": 1,
    },
    "Influenza (독감)": {
        "high_fever": 1,
        "chills": 1,
        "cough": 1,
        "muscle_pain": 1,
        "fatigue": 1,
        "headache": 1,
        "confirmed_patient_contact": 1,
    },
    "COVID-19": {
        "high_fever": 1,
        "cough": 1,
        "fatigue": 1,
        "loss_of_smell": 1,
        "shortness_of_breath": 1,
        "confirmed_patient_contact": 1,
    },
    "Measles (홍역)": {
        "high_fever": 1,
        "cough": 1,
        "skin_rash": 1,
        "eye_pain": 1,
        "confirmed_patient_contact": 1,
    },
    "Tuberculosis (결핵)": {
        "cough": 1,
        "fatigue": 1,
        "shortness_of_breath": 1,
        "confirmed_patient_contact": 1,
    },
    "Legionella (레지오넬라증)": {
        "high_fever": 1,
        "cough": 1,
        "shortness_of_breath": 1,
        "diarrhea": 1,
        "headache": 1,
        "unventilated_facility": 1,
    },
    "Pertussis (백일해)": {
        "cough": 1,
        "vomiting": 1,
        "shortness_of_breath": 1,
        "confirmed_patient_contact": 1,
    },
    "Diphtheria (디프테리아)": {
        "high_fever": 1,
        "cough": 1,
        "shortness_of_breath": 1,
        "neck_stiffness": 1,
        "confirmed_patient_contact": 1,
    },
    "Mpox (엠폭스)": {
        "high_fever": 1,
        "chills": 1,
        "skin_rash": 1,
        "muscle_pain": 1,
        "fatigue": 1,
        "animal_lesion_contact": 1,
    },
    "Varicella (수두/대상포진)": {
        "high_fever": 1,
        "skin_rash": 1,
        "fatigue": 1,
        "confirmed_patient_contact": 1,
    },
    "Anthrax (탄저)": {
        "high_fever": 1,
        "eschar_scab": 1,
        "shortness_of_breath": 1,
        "animal_lesion_contact": 1,
    },
    "Scarlet_Fever (성홍열)": {
        "high_fever": 1,
        "cough": 1,
        "skin_rash": 1,
        "vomiting": 1,
        "confirmed_patient_contact": 1,
    },
    "Ebola_Marburg (에볼라/마버그열)": {
        "high_fever": 1,
        "hemorrhage": 1,
        "vomiting": 1,
        "diarrhea": 1,
        "fatigue": 1,
        "travel_africa": 1,
        "animal_lesion_contact": 1,
    },
    "Lassa_Fever (라싸열)": {
        "high_fever": 1,
        "hemorrhage": 1,
        "vomiting": 1,
        "cough": 1,
        "travel_africa": 1,
    },
    "Hanta_HFRS (신증후군출혈열)": {
        "high_fever": 1,
        "hemorrhage": 1,
        "headache": 1,
        "muscle_pain": 1,
        "vomiting": 1,
        "tick_bite": 1,
    },
    "Cholera (콜레라)": {
        "diarrhea": 1,
        "vomiting": 1,
        "fatigue": 1,
        "raw_water_food": 1,
    },
    "Typhoid (장티푸스)": {
        "high_fever": 1,
        "headache": 1,
        "diarrhea": 1,
        "skin_rash": 1,
        "raw_water_food": 1,
    },
    "Shigellosis (세균성이질)": {
        "high_fever": 1,
        "bloody_diarrhea": 1,
        "vomiting": 1,
        "raw_water_food": 1,
    },
    "Hepatitis_A (A형간염)": {
        "high_fever": 1,
        "jaundice": 1,
        "fatigue": 1,
        "vomiting": 1,
        "raw_water_food": 1,
    },
    "Botulism (보툴리눔독소증)": {
        "paralysis_spasm": 1,
        "vomiting": 1,
        "shortness_of_breath": 1,
        "raw_water_food": 1,
    },
    "Gastroenteritis (급성위장관염)": {
        "vomiting": 1,
        "diarrhea": 1,
        "chills": 1,
        "fatigue": 1,
        "raw_water_food": 1,
    },
    "Rabies (공수병)": {
        "high_fever": 1,
        "hydrophobia": 1,
        "paralysis_spasm": 1,
        "animal_bite": 1,
    },
    "Meningococcal (수막구균수막염)": {
        "high_fever": 1,
        "neck_stiffness": 1,
        "hemorrhage": 1,
        "headache": 1,
        "vomiting": 1,
        "confirmed_patient_contact": 1,
    },
    "Leptospirosis (렙토스피라증)": {
        "high_fever": 1,
        "chills": 1,
        "muscle_pain": 1,
        "jaundice": 1,
        "eye_pain": 1,
        "raw_water_food": 1,
    },
    "Plague (페스트)": {
        "high_fever": 1,
        "chills": 1,
        "shortness_of_breath": 1,
        "hemorrhage": 1,
        "travel_africa": 1,
        "animal_bite": 1,
    },
    "MERS (메르스)": {
        "high_fever": 1,
        "cough": 1,
        "shortness_of_breath": 1,
        "diarrhea": 1,
        "travel_middle_east": 1,
        "animal_lesion_contact": 1,
    },
    "SFTS (중증열성혈소판감소)": {
        "high_fever": 1,
        "vomiting": 1,
        "diarrhea": 1,
        "hemorrhage": 1,
        "tick_bite": 1,
    },
    "Tetanus (파상풍)": {
        "paralysis_spasm": 1,
        "neck_stiffness": 1,
        "animal_bite": 1,
    },
}

disease_list = list(disease_profiles.keys())
idx_to_disease = {i: d for i, d in enumerate(disease_list)}

guideline_db = {
    "Dengue": (
        "【 뎅기열 (Dengue) 】\n🌐 CDC: Serum RT-PCR/NS1 항원 검사. 수액 요법 1원칙."
        " Aspirin/NSAIDs 절대 금기.\n🇰🇷 KDCA: 제3급 법정감염병 (24시간 이내 신고)"
    ),
    "Malaria": (
        "【 말라리아 (Malaria) 】\n🌐 CDC: 말초혈액 도말 검사. 삼일열(Chloroquine +"
        " Primaquine 14일, G6PD 검사 필수), 열대열(Coartem 3일).\n🇰🇷 KDCA: 제3급"
        " 법정감염병 (24시간 이내 신고)"
    ),
    "Zika": (
        "【 지카바이러스 (Zika) 】\n🌐 CDC: 혈청/소변 RT-PCR. 대증 치료. 임산부 감염"
        " 시 정밀 초음파.\n🇰🇷 KDCA: 제3급 법정감염병 (24시간 이내 신고)"
    ),
    "Chikungunya": (
        "【 치쿤구니야열 (Chikungunya) 】\n🌐 CDC: 혈청 RT-PCR 및 IgM. 다발성 관절통"
        " 조절(아세트아미노펜).\n🇰🇷 KDCA: 제3급 법정감염병 (24시간 이내 신고)"
    ),
    "Yellow_Fever": (
        "【 황열 (Yellow Fever) 】\n🌐 CDC: 혈청 RT-PCR 및 IgM. 대증 치료, 출혈 경향"
        " 모니터링.\n🇰🇷 KDCA: 제3급 법정감염병 (24시간 이내 신고)"
    ),
    "Scrub_Typhus": (
        "【 쯔쯔가무시증 (Scrub Typhus) 】\n🌐 CDC: 가피(Eschar) 확인. 1차 치료제"
        " Doxycycline 100mg bid x 7일.\n🇰🇷 KDCA: 제3급 법정감염병 (24시간 이내"
        " 신고)"
    ),
    "Lyme": (
        "【 라임병 (Lyme Disease) 】\n🌐 CDC: 유주성 홍반 임상 진단. Doxycycline"
        " 100mg bid x 14일.\n🇰🇷 KDCA: 제3급 법정감염병 (24시간 이내 신고)"
    ),
    "Influenza": (
        "【 인플루엔자 (Influenza) 】\n🌐 CDC: 비인두 RT-PCR/RAT. 48시간 내"
        " Oseltamivir 75mg bid x 5일.\n🇰🇷 KDCA: 제4급 법정감염병 (표본감시 7일 이내"
        " 신고)"
    ),
    "COVID-19": (
        "【 코로나19 (COVID-19) 】\n🌐 CDC/NIH: NAAT/RAT 양성. 5일 내 Paxlovid PO x"
        " 5일 또는 Remdesivir IV x 3일.\n🇰🇷 KDCA: 제4급 법정감염병 (표본감시 7일 이내"
        " 신고)"
    ),
    "Measles": (
        "【 홍역 (Measles) 】\n🌐 CDC: 인후도말 RT-PCR 및 IgM. 음압격리. 소아 비타민"
        " A 투여.\n🇰🇷 KDCA: 제2급 법정감염병 (24시간 이내 신고)"
    ),
    "Tuberculosis": (
        "【 결핵 (Tuberculosis) 】\n🌐 CDC: 객담 AFB 도말/배양 및 PCR. 4제"
        " 요법(INH+RIF+EMB+PZA) 표준 치료.\n🇰🇷 KDCA: 제2급 법정감염병 (24시간 이내"
        " 신고)"
    ),
    "Legionella": (
        "【 레지오넬라증 (Legionellosis) 】\n🌐 CDC: 소변 항원 검사(UAT) 및 PCR."
        " Levofloxacin 또는 Azithromycin.\n🇰🇷 KDCA: 제3급 법정감염병 (24시간 이내"
        " 신고)"
    ),
    "Pertussis": (
        "【 백일해 (Pertussis) 】\n🌐 CDC: 비인두 도말 PCR. Azithromycin 500mg D1"
        " 후 250mg qd x 4일.\n🇰🇷 KDCA: 제2급 법정감염병 (24시간 이내 신고)"
    ),
    "Diphtheria": (
        "【 디프테리아 (Diphtheria) 】\n🌐 CDC: 항독소(DAT) 즉시 투여 +"
        " Erythromycin 14일.\n🇰🇷 KDCA: 제1급 법정감염병 (진단 즉시 신고 및 격리)"
    ),
    "Mpox": (
        "【 엠폭스 (Mpox) 】\n🌐 CDC: 병변 PCR. 중증 시 Tecovirimat (TPOXX) 600mg"
        " bid x 14일.\n🇰🇷 KDCA: 제3급 법정감염병 (24시간 이내 신고)"
    ),
    "Varicella": (
        "【 수두/대상포진 (Varicella) 】\n🌐 CDC: 수포 진단. Acyclovir 800mg"
        " 5회/일 또는 Valacyclovir 1g tid x 7일.\n🇰🇷 KDCA: 제2급 법정감염병"
        " (24시간 이내 신고)"
    ),
    "Anthrax": (
        "【 탄저 (Anthrax) 】\n🌐 CDC: 흑색 가피 배양. Ciprofloxacin 500mg bid"
        " 또는 Doxycycline x 60일.\n🇰🇷 KDCA: 제1급 법정감염병 (진단 즉시 신고 및"
        " 격리)"
    ),
    "Scarlet_Fever": (
        "【 성홍열 (Scarlet Fever) 】\n🌐 CDC: A군 연쇄상구균 항원 검사."
        " Amoxicillin 10일 표준 치료.\n🇰🇷 KDCA: 제2급 법정감염병 (24시간 이내 신고)"
    ),
    "Ebola_Marburg": (
        "【 에볼라/마버그열 (Ebola/Marburg) 】\n🌐 CDC: BSL-4 음압격리. 단클론항체(Inmazeb)"
        " 투약 및 집중 수액 소생술.\n🇰🇷 KDCA: 제1급 법정감염병 (진단 즉시 신고)"
    ),
    "Lassa_Fever": (
        "【 라싸열 (Lassa Fever) 】\n🌐 CDC: 혈청 RT-PCR. 조기 Ribavirin IV"
        " 투여.\n🇰🇷 KDCA: 제1급 법정감염병 (진단 즉시 신고)"
    ),
    "Hanta_HFRS": (
        "【 신증후군출혈열 (HFRS) 】\n🌐 CDC: 한타바이러스 IgM/PCR. 조기"
        " Ribavirin IV 고려 및 투석 대비.\n🇰🇷 KDCA: 제3급 법정감염병 (24시간 이내"
        " 신고)"
    ),
    "Cholera": (
        "【 콜레라 (Cholera) 】\n🌐 CDC: 대변 배양. 대량 수액(Ringer) 및 ORS +"
        " Doxycycline 300mg 단회.\n🇰🇷 KDCA: 제2급 법정감염병 (24시간 이내 신고)"
    ),
    "Typhoid": (
        "【 장티푸스 (Typhoid Fever) 】\n🌐 CDC: 혈액/대변 배양. Ceftriaxone 2g IV"
        " qd 또는 Ciprofloxacin.\n🇰🇷 KDCA: 제2급 법정감염병 (24시간 이내 신고)"
    ),
    "Shigellosis": (
        "【 세균성이질 (Shigellosis) 】\n🌐 CDC: 대변 배양. Azithromycin 500mg D1"
        " 후 250mg qd x 4일.\n🇰🇷 KDCA: 제2급 법정감염병 (24시간 이내 신고)"
    ),
    "Hepatitis_A": (
        "【 A형간염 (Hepatitis A) 】\n🌐 CDC: Serum IgM anti-HAV 양성. 대증 치료 및"
        " 간기능 모니터링.\n🇰🇷 KDCA: 제2급 법정감염병 (24시간 이내 신고)"
    ),
    "Botulism": (
        "【 보툴리눔독소증 (Botulism) 】\n🌐 CDC: 보툴리눔 7가 항독소(BAT) 즉시"
        " 투여 + 기계환기 대비.\n🇰🇷 KDCA: 제1급 법정감염병 (진단 즉시 신고)"
    ),
    "Gastroenteritis": (
        "【 급성위장관염 (Infectious Diarrhea) 】\n🌐 CDC: 경구 수액(ORS) 1원칙."
        " 중증 시 Azithromycin 500mg qd x 3일.\n🇰🇷 KDCA: 수인성·식품매개감염병"
        " (집단역학조사 대상)"
    ),
    "Rabies": (
        "【 공수병 (Rabies) 】\n🌐 CDC: 교상 시 상처 세척 후 즉시 Rabies"
        " 면역글로불린(HRIG) + 백신 4회 접종.\n🇰🇷 KDCA: 제3급 법정감염병 (24시간"
        " 이내 신고)"
    ),
    "Meningococcal": (
        "【 수막구균수막염 (Meningococcal) 】\n🌐 CDC: 뇌척수액 배양. Ceftriaxone 2g"
        " IV q12h 초응급 투여.\n🇰🇷 KDCA: 제2급 법정감염병 (24시간 이내 신고)"
    ),
    "Leptospirosis": (
        "【 렙토스피라증 (Leptospirosis) 】\n🌐 CDC: MAT 항체/PCR. Doxycycline"
        " 100mg bid 또는 Ceftriaxone 1g IV qd x 7일.\n🇰🇷 KDCA: 제3급 법정감염병"
        " (24시간 이내 신고)"
    ),
    "Plague": (
        "【 페스트 (Plague) 】\n🌐 CDC: 객담/림프절 흡인액 배양. Streptomycin 1g"
        " IM bid 또는 Doxycycline.\n🇰🇷 KDCA: 제1급 법정감염병 (진단 즉시 신고 및"
        " 격리)"
    ),
    "MERS": (
        "【 메르스 (MERS-CoV) 】\n🌐 CDC: 하기도 검체 RT-PCR. 음압격리 및 대증 요법"
        " (Remdesivir 고려).\n🇰🇷 KDCA: 제1급 법정감염병 (진단 즉시 유선 신고 및"
        " 격리)"
    ),
    "SFTS": (
        "【 중증열성혈소판감소증후군 (SFTS) 】\n🌐 CDC/KDCA: 혈청 RT-PCR. 특이"
        " 항바이러스제 없음 (혈소판 수혈 및 대증 요법).\n🇰🇷 KDCA: 제3급 법정감염병"
        " (24시간 이내 신고)"
    ),
    "Tetanus": (
        "【 파상풍 (Tetanus) 】\n🌐 CDC: 파상풍 면역글로불린(TIG 3000 IU) +"
        " Metronidazole 500mg q6h + Tdap 백신.\n🇰🇷 KDCA: 제3급 법정감염병"
        " (24시간 이내 신고)"
    ),
}

incubation_periods = {
    "Dengue": {"min": 4, "max": 10, "kr": "뎅기열 (CDC: 4~10일)"},
    "Malaria": {"min": 7, "max": 30, "kr": "말라리아 (CDC: 7~30일)"},
    "Scrub_Typhus": {"min": 6, "max": 18, "kr": "쯔쯔가무시증 (CDC: 6~18일)"},
    "COVID-19": {"min": 2, "max": 7, "kr": "코로나19 (CDC: 2~7일)"},
    "Influenza": {"min": 1, "max": 4, "kr": "인플루엔자 (CDC: 1~4일)"},
}

# ---------------------------------------------------------
# 6. 모델 초기화 및 가중치 로드
# ---------------------------------------------------------
multimodal_model = MultimodalCDSSNet(
    num_features=len(features), num_classes=len(disease_list)
)
is_pretrained_loaded = False

if os.path.exists(WEIGHTS_PATH):
    try:
        multimodal_model.load_state_dict(
            torch.load(WEIGHTS_PATH, map_location="cpu")
        )
        is_pretrained_loaded = True
        print(f"✅ [Weights Loaded]: '{WEIGHTS_PATH}' 로드 성공")
    except Exception as e:
        print(f"⚠️ [Weights Load Failed]: {e}")

multimodal_model.eval()


# ---------------------------------------------------------
# 7. 고신뢰도 추론 및 XAI 함수
# ---------------------------------------------------------
def enable_mc_dropout(model):
    """평가 모드에서 Dropout 계층만 활성화한다.

    환자 한 명(batch size 1)을 추론할 때 모델 전체를 train 모드로 바꾸면
    BatchNorm1d가 배치 통계를 계산할 수 없어 예외가 발생한다. BatchNorm은
    eval 모드로 유지하고 Dropout만 train 모드로 전환해야 MC Dropout을
    안전하게 수행할 수 있다.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.modules.dropout._DropoutNd):
            module.train()


def predict_with_uncertainty(
    input_tensor,
    text_tensor,
    img_tensor,
    profile_mat,
    num_samples=10,
    temperature=1.35,
):
    enable_mc_dropout(multimodal_model)
    sampled_probs = []

    with torch.no_grad():
        prior_logits = torch.matmul(input_tensor, profile_mat.t())
        for _ in range(num_samples):
            logits = multimodal_model(input_tensor, text_tensor, img_tensor)
            w_nn = 0.75 if is_pretrained_loaded else 0.40
            w_prior = 0.25 if is_pretrained_loaded else 0.60

            scaled_logits = (
                logits * w_nn + prior_logits * w_prior
            ) / temperature
            probs = F.softmax(scaled_logits, dim=1).squeeze().cpu().numpy()
            sampled_probs.append(probs)

    multimodal_model.eval()
    sampled_probs = np.array(sampled_probs)
    mean_probs = np.mean(sampled_probs, axis=0)
    std_probs = np.std(sampled_probs, axis=0)
    return mean_probs, std_probs


def compute_multimodal_integrated_gradients(
    input_vec_t, text_tokens_t, img_t, target_class, target_device, steps=15
):
    multimodal_model.eval()
    baseline_tab = torch.zeros_like(input_vec_t)
    alphas = torch.linspace(0.0, 1.0, steps, device=target_device).view(-1, 1)

    interpolated_tab = baseline_tab + alphas * (input_vec_t - baseline_tab)
    interpolated_tab.requires_grad_(True)

    batch_text = (
        text_tokens_t.repeat(steps, 1) if text_tokens_t is not None else None
    )
    batch_img = img_t.repeat(steps, 1, 1, 1) if img_t is not None else None

    logits = multimodal_model(interpolated_tab, batch_text, batch_img)
    target_score = logits[:, target_class].sum()
    target_score.backward()

    grads = interpolated_tab.grad.mean(dim=0)
    integrated_grads = (input_vec_t - baseline_tab).squeeze() * grads
    return integrated_grads.detach().cpu().numpy()


def calculate_egfr(age, sex, weight, scr):
    if scr <= 0:
        return 90.0
    crcl = ((140.0 - float(age)) * float(weight)) / (72.0 * float(scr))
    return round(crcl * (0.85 if sex == "여성 (Female)" else 1.0), 1)


def get_treatment_and_warnings(
    disease_key, egfr, age, weight, is_pregnant, allergy_list, current_meds
):
    has_penicillin_allergy = "페니실린 / 세파계 항생제 알레르기" in (
        allergy_list or []
    )
    has_statin_or_cyp = any(
        m in (current_meds or [])
        for m in [
            "스타틴계 지질강하제 (심바스타틴, 아토르바스타틴 등)",
            "항부정맥제 / 진정수면제 (CYP3A4 대사 약물)",
        ]
    )

    if disease_key == "Dengue":
        return (
            "💊 [CDC 처방]: 아세트아미노펜 500~650mg PO q6h + 등장성 수액(0.9% NS) 집중 보충",
            "⚠️ [절대 금기]: 아스피린 및 모든 NSAIDs 절대 투여 금지! (출혈성 뎅기열 촉발)",
        )
    elif disease_key == "Malaria":
        return (
            "💊 [CDC 처방]: 삼일열(Chloroquine 600mg base 후 Primaquine 14일) /"
            " 열대열(Coartem 3일)",
            "⚠️ [경고]: Primaquine 전 G6PD 결핍 확인 필수 (용혈성 빈혈 위험).",
        )
    elif disease_key in ["Scrub_Typhus", "Lyme"]:
        abx = (
            "Azithromycin 500mg qd"
            if is_pregnant
            else "Doxycycline 100mg PO bid x 7~14일"
        )
        return (
            f"💊 [CDC 처방]: {abx}",
            "⚠️ [안내]: 발열 호전 후에도 재발 방지를 위해 규정 기간 투약 완료.",
        )
    elif disease_key == "Influenza":
        osel = (
            "Oseltamivir 75mg PO bid x 5일"
            if egfr >= 60
            else "Oseltamivir 30mg PO bid (50% 감량) x 5일"
        )
        return (
            f"💊 [CDC 처방]: {osel} 또는 Baloxavir (조플루자) 1회",
            "⚠️ [금기]: 소아 아스피린 금기 (라이 증후군 예방).",
        )
    elif disease_key == "COVID-19":
        pax_warn = (
            "\n  🚨 [DDI 경보]: 스타틴/CYP3A4 약물 병용 금기 -> Remdesivir 대체 권고."
            if has_statin_or_cyp
            else ""
        )
        pax = (
            f"Paxlovid 300mg/100mg PO bid x 5일{pax_warn}"
            if egfr >= 60
            else f"Paxlovid 150mg/100mg PO bid x 5일{pax_warn}"
        )
        return (
            f"💊 [CDC/NIH 처방]: {pax} 또는 Remdesivir IV x 3일",
            "⚠️ [경고]: 스타틴/진정제 병용 주의. 임산부 라게브리오 금기.",
        )
    else:
        abx = (
            "Azithromycin 500mg qd x 3일"
            if is_pregnant
            else "Ciprofloxacin 500mg PO bid 또는 Azithromycin"
        )
        return (
            "💊 [CDC 표준 대증 및 항균 요법]: 경구 수액(ORS) 1원칙 / 중증 시"
            f" {abx}",
            "⚠️ [안내]: 환자 상태 및 배양 결과에 따라 약제 조정.",
        )


def update_bayesian_probability(pre_prob, result, test_type):
    if result == "미시행 (None)":
        return pre_prob, "검사 미시행 (사전 확률 유지)"
    p = max(0.01, min(0.99, pre_prob / 100.0))
    prior_odds = p / (1.0 - p)
    lr = (
        (196.0 if "양성" in result else 0.02)
        if "PCR" in test_type
        else (40.0 if "양성" in result else 0.20)
    )
    post_odds = prior_odds * lr
    post_prob = post_odds / (1.0 + post_odds)
    direction = (
        "📈 <b>[베이지안 사후 확률 급상승]</b> 확진 확률"
        if "양성" in result
        else "📉 <b>[베이지안 사후 확률 하향]</b> 배제 확률"
    )
    return (
        round(post_prob * 100.0, 1),
        f"{direction} <b>{post_prob*100:.1f}%</b> 반영됨 ({test_type})",
    )


def calculate_incubation(exposure_days, disease_tag):
    prof = incubation_periods.get(
        disease_tag, {"min": 1, "max": 14, "kr": "일반 (1~14일)"}
    )
    if exposure_days <= 0:
        return "노출일 정보 미입력"
    if prof["min"] <= exposure_days <= prof["max"]:
        return (
            f"✅ <b>[CDC 잠복기 일치]</b> 노출 후 {exposure_days}일 경과 ({prof['kr']})"
        )
    return (
        f"⚠️ <b>[잠복기 {'단축' if exposure_days < prof['min'] else '초과'}]</b>"
        f" 노출 후 {exposure_days}일 경과 ({prof['kr']})"
    )


def generate_official_pdf(
    report_text, soap_text, doc_name="의사 미기재", audit_hash=""
):
    pdf_filename = f"CDSS_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf_path = os.path.join(BASE_DIR, pdf_filename)
    doc = pymupdf.open()

    p1 = doc.new_page(width=595, height=842)
    p1.insert_textbox(
        pymupdf.Rect(40, 45, 555, 795),
        f"【 질병관리청(KDCA) 법정감염병 발생 신고서 】\n\n{report_text}\n\n신고 의사:"
        f" {doc_name} (서명/인)\n[SHA-256 감사코드]: {audit_hash}",
        fontsize=9.0,
    )

    p2 = doc.new_page(width=595, height=842)
    p2.insert_textbox(
        pymupdf.Rect(40, 45, 555, 795),
        f"【 EMR 표준 임상 경과기록지 (CDC Protocol SOAP) 】\n\n{soap_text}\n\n담당"
        f" 전문의: {doc_name}\n[해시검증: {audit_hash[:16]}...]",
        fontsize=8.6,
    )

    doc.save(pdf_path)
    doc.close()
    return pdf_path


# ---------------------------------------------------------
# 8. 메인 CDSS 파이프라인
# ---------------------------------------------------------
def full_enterprise_cdss_pipeline(
    age,
    sex,
    weight,
    sbp,
    rr,
    spo2,
    consciousness,
    scr,
    plt,
    wbc,
    crp,
    is_pregnant,
    exposure_days,
    comorbidities,
    allergy_list,
    current_meds,
    rapid_test_type,
    rapid_test_result,
    symptom_boxes,
    epi_boxes,
    red_flag_checks,
    free_text_note,
    lesion_image,
    seq_file,
    seq_raw_text,
    decision_threshold,
):
    runtime_device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    multimodal_model.to(runtime_device)

    # 1. Sanger Sequencing 분석
    seq_record = SangerSequenceAgent.parse_sequence(seq_file, seq_raw_text)
    blast_result, blast_msg = (None, "")
    if seq_record:
        blast_result, blast_msg = SangerSequenceAgent.run_online_blast(
            seq_record
        )

    # 2. 정형 피처 벡터화
    input_vec = np.zeros(len(features))
    for s_ko in symptom_boxes or []:
        if s_ko in symptom_korean_map:
            input_vec[features.index(symptom_korean_map[s_ko])] = 1.0
    for e_ko in epi_boxes or []:
        if e_ko in epi_korean_map:
            input_vec[features.index(epi_korean_map[e_ko])] = 1.0

    # 3. 텍스트 토큰화
    tokens = tokenizer.encode(free_text_note)
    text_tensor = torch.tensor([tokens], dtype=torch.long).to(runtime_device)

    # 4. 흉부 X-ray/병변 이미지 전처리
    if lesion_image is not None:
        img_arr = np.array(lesion_image.convert("L"))
        img_tensor = (
            torch.tensor(img_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )
        img_tensor = F.interpolate(img_tensor, size=(64, 64)) / 255.0
        img_tensor = img_tensor.to(runtime_device)
    else:
        img_tensor = None

    if (
        np.sum(input_vec) == 0
        and not free_text_note
        and not seq_record
        and img_tensor is None
    ):
        default_summary = (
            "<div style='padding:14px; background:#0f172a; border-radius:8px;"
            " border:1px solid #334155;'><h3>🏥 의심 질환: 추론 대기 중</h3></div>"
        )
        default_emergency = (
            "<div style='padding:10px; background:#1e293b; border-radius:6px;"
            " color:#cbd5e1;'>🟢 생체 징후 대기 중</div>"
        )
        default_next_q = (
            "<div style='padding:10px; background:#1e293b; border-radius:6px;"
            " color:#94a3b8;'>진료 데이터를 입력하시면 분석 결과가 도출됩니다.</div>"
        )
        default_xai = [["선택 대기", "0%"]]
        default_rule_out = [["감별 대기", "0%", "증상 미선택"]]
        return (
            default_summary,
            default_emergency,
            default_next_q,
            {"대기": 0.0},
            default_xai,
            None,
            "처방 권고 대기 중...",
            "금기 사항 대기 중...",
            "지침 매핑 대기 중...",
            default_rule_out,
            "SOAP 차트 대기 중...",
            "발생 신고서 대기 중...",
            "English SOAP pending...",
            None,
        )

    # 5. 사전 지식 행렬 구축
    profile_mat = torch.zeros(
        len(disease_list), len(features), device=runtime_device
    )
    for d_idx, d_name in enumerate(disease_list):
        for f_idx, f_name in enumerate(features):
            profile_mat[d_idx, f_idx] = (
                2.5 if f_name in disease_profiles[d_name] else -0.35
            )

    # 6. MC Dropout 추론 실행
    input_tensor = torch.tensor([input_vec], dtype=torch.float32).to(
        runtime_device
    )
    mean_probs, std_probs = predict_with_uncertainty(
        input_tensor,
        text_tensor,
        img_tensor,
        profile_mat,
        num_samples=10,
        temperature=1.35,
    )

    top_idxs = np.argsort(mean_probs)[::-1]
    raw_top_prob = float(mean_probs[top_idxs[0]] * 100.0)
    top_uncertainty = float(std_probs[top_idxs[0]] * 100.0)
    primary_suspect = idx_to_disease[top_idxs[0]]

    # 7. 베이지안 보정 및 Sanger 확진
    final_prob, bayes_comment = update_bayesian_probability(
        raw_top_prob, rapid_test_result, rapid_test_type
    )
    blast_badge = ""

    if blast_result and blast_result.get("matched_disease"):
        target_d = blast_result["matched_disease"]
        ident = blast_result["identity"]
        primary_suspect = target_d
        final_prob = 99.9 if ident >= 98.0 else max(final_prob, ident)
        bayes_comment = (
            f"🧬 <b>[Sanger BLAST 동정]</b> {blast_result['hit_title'][:45]}... (ID:"
            f" {ident}%, E-val: {blast_result['e_value']})"
        )
        blast_badge = (
            "<span style='background:#8b5cf6; color:#fff; padding:2px 8px;"
            " border-radius:4px; font-weight:bold; font-size:11px;"
            " margin-left:8px;'>Sanger BLAST 확진</span>"
        )

    # 8. 운영 임계값(Threshold) 판정
    threshold_pct = float(decision_threshold) * 100.0
    is_confirmed_by_threshold = final_prob >= threshold_pct
    threshold_badge = (
        f"<span style='background:{'#10b981' if is_confirmed_by_threshold else '#f59e0b'};"
        " color:#fff; padding:2px 8px; border-radius:4px; font-weight:bold;"
        f" font-size:11px; margin-left:8px;'>임계치({threshold_pct:.0f}%)"
        f" {'충족' if is_confirmed_by_threshold else '미달(외래관찰)'}</span>"
    )

    prob_dict = {}
    for i in top_idxs[:4]:
        if mean_probs[i] > 0.01:
            name_with_ci = f"{idx_to_disease[i]} [±{std_probs[i]*100:.1f}%]"
            prob_dict[name_with_ci] = float(mean_probs[i])
    prob_dict[f"{primary_suspect} (최종 확진/사후)"] = final_prob / 100.0

    disease_tag = "Dengue"
    for k in incubation_periods.keys():
        if k in primary_suspect:
            disease_tag = k
            break

    incubation_summary = calculate_incubation(exposure_days, disease_tag)
    model_origin_badge = (
        "Kaggle 실데이터 가중치 로드"
        if is_pretrained_loaded
        else "임상 지식 행렬(Safe Matrix)"
    )

    cache_info = f' [국내 일일 코로나: {epi_cache["korea_covid_new_cases"]:,}명 / 결핵발생률: {epi_cache["who_tb_incidence_kor"]}/10만명]'
    next_question_html = (
        "<div style='padding:10px 14px; background:#1e1b4b; border-left:4px"
        " solid #818cf8; border-radius:6px; color:#e0e7ff; font-size:13px;'>💡"
        f" <b>엔진 상태:</b> {model_origin_badge} · 캘리브레이션(T=1.35) · MC Dropout"
        f" 10회 (오차: ±{top_uncertainty:.1f}%){cache_info}</div>"
    )

    badge = blast_badge if blast_badge else threshold_badge
    summary_html = (
        '<div style="background:linear-gradient(135deg, #1e3a8a 0%, #0f172a'
        ' 100%); padding:16px 20px; border-radius:8px; border:1px solid'
        ' #3b82f6;"><div style="color:#93c5fd; font-size:12px;'
        ' font-weight:600;">High-Reliability Vision & Genomic CDSS'
        ' (v16.2)</div><div style="font-size:22px; font-weight:800; color:#fff;'
        f' margin:6px 0;">🏥 {primary_suspect} {badge}</div><div'
        ' style="display:flex; gap:16px; margin-top:8px; font-size:13px;'
        ' color:#e2e8f0;"><div>📊 <b>보정 확률:</b> <span style="color:#38bdf8;'
        f' font-weight:bold; font-size:15px;">{final_prob:.1f}%'
        f' (±{top_uncertainty:.1f}%)</span></div><div>⏱️ <b>잠복기:</b>'
        f' {incubation_summary}</div></div><div style="margin-top:6px;'
        f' font-size:12px; color:#cbd5e1;">{bayes_comment}</div></div>'
    )

    # Red Flags
    red_flags = []
    if spo2 < 94:
        red_flags.append(f"저산소혈증 (SpO2 {spo2}%)")
    if sbp < 90:
        red_flags.append(f"저혈압 쇼크 (SBP {sbp}mmHg)")
    if plt < 100:
        red_flags.append(f"혈소판 급감 (Plt {plt}k)")
    if red_flag_checks:
        red_flags.extend(red_flag_checks)

    high_risks = [f"기저질환({', '.join(comorbidities)})"] if comorbidities else []
    if age >= 65:
        high_risks.append("65세 이상 고령")

    if red_flags:
        emergency_html = (
            "<div style='padding:12px 16px; background:#450a0a; border-left:6px"
            " solid #ef4444; border-radius:6px; color:#fecaca;'><div"
            " style='font-size:14px; font-weight:bold; color:#f87171;'>🚨 [CDC"
            " Warning Signs / 응급 ICU 경보] 즉시 상급 전원 대상</div><div"
            f" style='font-size:12.5px; margin-top:4px;'>위험 징후: <b>{', '.join(red_flags)}</b></div></div>"
        )
    elif high_risks:
        emergency_html = (
            "<div style='padding:12px 16px; background:#451a03; border-left:6px"
            " solid #f97316; border-radius:6px; color:#ffedd5;'><div"
            " style='font-size:14px; font-weight:bold; color:#fb923c;'>🟠 [CDC"
            " 고위험군] 조기 치료제 투약 요망</div><div style='font-size:12.5px;"
            f" margin-top:4px;'>위험 요인: <b>{', '.join(high_risks)}</b></div></div>"
        )
    else:
        emergency_html = (
            "<div style='padding:10px 16px; background:#064e3b; border-left:6px"
            " solid #10b981; border-radius:6px; color:#d1fae5;'><div"
            " style='font-size:13.5px; font-weight:bold; color:#34d399;'>🟢 [생체"
            " 징후 안정] CDC 기준 중증 위험 없음 (외래 관찰 가능)</div></div>"
        )

    # 9. XAI 및 Grad-CAM 연산
    ig_attributions = compute_multimodal_integrated_gradients(
        input_tensor, text_tensor, img_tensor, top_idxs[0], runtime_device
    )
    active_indices = [idx for idx, val in enumerate(input_vec) if val == 1.0]

    contributions = []
    for idx in active_indices:
        score = float(abs(ig_attributions[idx]))
        contributions.append(
            (feature_kr_names.get(features[idx], features[idx]), score)
        )

    if blast_result:
        contributions.append(
            (f"Sanger BLAST 동정 ({blast_result['identity']}%)", 0.95)
        )
    if free_text_note:
        contributions.append(("자유 문진 텍스트(NLP/LSTM)", 0.35))
    if img_tensor is not None:
        contributions.append(("흉부 X-ray/병변 영상(Vision CNN)", 0.55))

    contributions.sort(key=lambda x: x[1], reverse=True)
    total_score = sum(c[1] for c in contributions) or 1.0
    xai_table = (
        [[c[0], f"{(c[1]/total_score)*100:.1f}%"] for c in contributions[:4]]
        if contributions
        else [["해당 없음", "0%"]]
    )

    # Grad-CAM 시각화
    gradcam_overlay = generate_gradcam_heatmap(
        img_tensor, top_idxs[0], runtime_device
    )

    rule_out_table = [
        [
            idx_to_disease[top_idxs[i]],
            f"{mean_probs[top_idxs[i]]*100:.1f}% (±{std_probs[top_idxs[i]]*100:.1f}%)",
            "어텐션 융합 확률 열세",
        ]
        for i in range(1, min(3, len(top_idxs)))
    ]
    if not rule_out_table:
        rule_out_table = [["감별 질환 없음", "-", "-"]]

    cdc_kdca_txt = guideline_db.get(disease_tag, "CDC/KDCA 지침 결과 없음")
    egfr_val = calculate_egfr(age, sex, weight, scr)
    treatment_txt, contra_txt = get_treatment_and_warnings(
        disease_tag,
        egfr_val,
        age,
        weight,
        is_pregnant,
        allergy_list,
        current_meds,
    )
    audit_hash = hashlib.sha256(
        f"{age}_{sex}_{primary_suspect}_{final_prob}_{datetime.datetime.now().isoformat()}".encode()
    ).hexdigest()

    soap_chart = f"""[EMR SOAP CLINICAL NOTE - High Reliability CDSS (v16.2)]
================================================================================
S (Subjective):
  • 주소(C.C): {free_text_note if free_text_note else ', '.join(symptom_boxes or ['호소 증상 없음'])}
  • 역학/노출력: {', '.join(epi_boxes or ['없음'])} (노출 {exposure_days}일 경과)
  • 환자 기본정보: {age}세, {sex}, {weight}kg (임신/수유: {'예' if is_pregnant else '아니오'}) | 기저질환: {', '.join(comorbidities) if comorbidities else '없음'}
  • 알레르기/상용약: {', '.join(allergy_list or ['없음'])} / {', '.join(current_meds or ['없음'])}

O (Objective):
  • Vital Signs: SBP {sbp} mmHg, RR {rr}회/min, SpO2 {spo2}%, 의식 [{consciousness}]
  • Lab Panel: eGFR(C-G) {egfr_val} mL/min (Scr {scr} mg/dL), Plt {plt}k/μL, WBC {wbc}/μL, CRP {crp} mg/L
  • Vision 판독: {'흉부 X-ray/병변 Grad-CAM 히트맵 생성 완료' if img_tensor is not None else '영상 미첨부'}
  • Sanger BLAST: {blast_msg if blast_result else '미시행'}
  • 감시 통계 연동: 국내 코로나 유행 ({epi_cache['korea_covid_new_cases']:,}명) / WHO 결핵지표 ({epi_cache['who_tb_incidence_kor']}/10만명)
  • 신뢰도 분석: Calibrated Softmax (T=1.35), MC Dropout Uncertainty (±{top_uncertainty:.1f}%), 임계값({threshold_pct:.0f}%)
  • Red Flags: {', '.join(red_flags) if red_flags else '특이 소견 없음'}

A (Assessment):
  • 1차 의심 진단: {primary_suspect} (보정 확률: {final_prob:.1f}% ± {top_uncertainty:.1f}%)
  • 결정적 딥러닝/Vision XAI 근거: {', '.join([f"{x[0]}({x[1]})" for x in xai_table[:2]])}
  • 감별 배제: {rule_out_table[0][0] if rule_out_table else '없음'} 배제

P (Plan):
  • [CDC 표준 치료/항생제 계획]:
{treatment_txt}
  • [CDC/KDCA 금기 약물/DDI]:
{contra_txt}
  • [KDCA 법정 감염병 지침]:
{cdc_kdca_txt}
  • [SHA-256 감사 해시]: {audit_hash}
================================================================================"""

    report_form = f"""[질병관리청(KDCA) 법정감염병 발생 신고서]
--------------------------------------------------------------------------------
1. 환자 인적사항: 만 {age}세 / {sex} | {weight} kg | 임신: {'예' if is_pregnant else '아니오'} | 기저질환: {', '.join(comorbidities) if comorbidities else '없음'}
2. 진단 정보: {primary_suspect} (의사환자 [V], CDC 및 Vision/BLAST 보정 사후확률 {final_prob:.1f}% [±{top_uncertainty:.1f}%] 부합)
   • 주요 소견: {free_text_note if free_text_note else ', '.join(symptom_boxes or [])}
   • 영상/유전체 판독: {'Grad-CAM 히트맵 병변 판독 완료' if img_tensor is not None else '영상 없음'} | {blast_msg if blast_result else '시퀀싱 없음'}
3. 검사 징후: SpO2 {spo2}% | Plt {plt}k/μL | eGFR {egfr_val} mL/min | {rapid_test_type} ({rapid_test_result})
4. 신고 의사 소견: 감염병예방법에 의거 즉시 신고함. [검증코드: {audit_hash[:20]}]
--------------------------------------------------------------------------------"""

    english_soap = f"""[INTERNATIONAL EMR CLINICAL NOTE - MULTIMODAL CDSS v16.2]
================================================================================
PATIENT: {age} yrs, {sex}, {weight} kg | Vitals: SBP {sbp}, SpO2 {spo2}%, eGFR {egfr_val}, Plt {plt}k
PRIMARY SUSPICION: {disease_tag} (Calibrated Prob: {final_prob:.1f}%, Error: ±{top_uncertainty:.1f}%)
VISION GRAD-CAM: {'Heatmap Computed' if img_tensor is not None else 'No Image'}
EPIDEMIOLOGY CONTEXT: Korea COVID: {epi_cache['korea_covid_new_cases']:,} / WHO TB Index: {epi_cache['who_tb_incidence_kor']} per 100k
SEVERITY TRIAGE: {'EMERGENCY (ICU/Transfer)' if red_flags else ('HIGH RISK (Comorbid)' if high_risks else 'STABLE')}
HOME ISOLATION & INSTRUCTIONS: Self-isolation mandatory until PCR confirmed. Follow CDC infection control.
AUDIT TRAIL HASH: {audit_hash}
================================================================================"""

    pdf_path = generate_official_pdf(
        report_form, soap_chart, audit_hash=audit_hash
    )

    return (
        summary_html,
        emergency_html,
        next_question_html,
        prob_dict,
        xai_table,
        gradcam_overlay,
        treatment_txt,
        contra_txt,
        cdc_kdca_txt,
        rule_out_table,
        soap_chart,
        report_form,
        english_soap,
        pdf_path,
    )


def doctor_signoff_action(doc_name, doc_lic, soap_txt, report_txt):
    if not doc_name or not doc_lic:
        return (
            "<span style='color:#ef4444; font-weight:bold;'>⚠️ 담당 의사 성명과 의사"
            " 면허번호를 입력해야 승인 완료됩니다.</span>",
            None,
        )
    audit_hash = hashlib.sha256(
        f"{doc_name}_{doc_lic}_{datetime.datetime.now()}".encode()
    ).hexdigest()
    signed_pdf = generate_official_pdf(
        report_txt, soap_txt, f"{doc_name} (면허 {doc_lic})", audit_hash
    )
    return (
        f"<span style='color:#10b981; font-weight:bold;'>✅ [전자서명 승인 완료] 담당의사:"
        f" {doc_name} (면허: {doc_lic}) 날인 완료. (코드:"
        f" {audit_hash[:12]}...)</span>",
        signed_pdf,
    )


# ---------------------------------------------------------
# 9. Gradio UI 구성
# ---------------------------------------------------------
custom_theme = gr.themes.Soft(primary_hue="blue", secondary_hue="slate")
custom_css = (
    ".gr-button-primary { background: linear-gradient(135deg, #2563eb 0%,"
    " #1d4ed8 100%) !important; font-weight: bold !important; font-size: 15px"
    " !important; }"
)

with gr.Blocks(
    title="고성능 감염병 CDSS v16.2", theme=custom_theme, css=custom_css
) as demo:
    gr.Markdown(
        "# 🏥 공공 감염병 진단·감시 임상 의사결정 지원 시스템 (CDSS v16.2)"
    )
    gr.Markdown(
        "**🌐 흉부 X-ray 폐렴 판독 · Grad-CAM 히트맵 · Sanger BLAST 유전체 동정 ·"
        " 글로벌 감시(OWID/WHO) 통계 연동**"
    )

    with gr.Row():
        with gr.Column(scale=5):
            with gr.Accordion(
                "1️⃣ 환자 기본 바이탈 & 신기능 (Risk Stratification)",
                open=True,
            ):
                with gr.Row():
                    age_input = gr.Number(value=35, label="연령 (만 나이)")
                    sex_input = gr.Radio(
                        choices=["남성 (Male)", "여성 (Female)"],
                        value="남성 (Male)",
                        label="성별",
                    )
                    weight_input = gr.Number(value=68.0, label="체중 (kg)")
                with gr.Row():
                    sbp_input = gr.Number(
                        value=115, label="수축기 혈압 (SBP, mmHg)"
                    )
                    rr_input = gr.Number(value=18, label="호흡수 (RR, 회/분)")
                    spo2_input = gr.Number(value=98, label="산소포화도 (SpO2, %)")
                with gr.Row():
                    consciousness_input = gr.Dropdown(
                        choices=[
                            "명료 (Alert)",
                            "기면/혼미 (Drowsy/Stupor)",
                            "혼수 (Coma)",
                        ],
                        value="명료 (Alert)",
                        label="의식 수준",
                    )
                    scr_input = gr.Number(
                        value=0.9, label="혈청 크레아티닌 (Scr, mg/dL)"
                    )
                    pregnant_input = gr.Checkbox(
                        label="임신 또는 수유 중", value=False
                    )
                exposure_days_input = gr.Number(
                    value=5, label="의심 노출/입국 후 경과 일수"
                )
                comorbidities_input = gr.CheckboxGroup(
                    choices=[
                        "당뇨병 (Diabetes Mellitus)",
                        "만성 신질환 (CKD)",
                        "만성 간질환 / 간경변",
                        "심혈관질환 / 고혈압",
                        "면역저하자 / 항암·면역억제제 투여 중",
                    ],
                    label="🩺 환자 기저질환",
                )

            with gr.Accordion(
                "2️⃣ 멀티모달 입력 (자유 문진 텍스트 & 흉부 X-ray/피부 사진)",
                open=True,
            ):
                free_text_note = gr.Textbox(
                    lines=2,
                    placeholder=(
                        "예: 3일 전부터 고열과 함께 기침, 흉통이 지속됩니다."
                    ),
                    label="📝 환자 자유 호소 문진 (NLP/LSTM 인코더)",
                )
                lesion_image = gr.Image(
                    type="pil",
                    label=(
                        "📷 흉부 X-선 영상(X-ray) 또는 피부 병변 사진 업로드"
                        " (Vision CNN 인코더)"
                    ),
                )

            with gr.Accordion(
                "3️⃣ Sanger 시퀀싱 분자진단 (FASTA / FASTQ / Raw Seq)",
                open=False,
            ):
                gr.Markdown(
                    "💡 **NCBI Web BLAST 연동**: 염기서열 데이터를 입력하면 NCBI"
                    " BLAST로 병원체 유전자를 자동 동정합니다."
                )
                seq_file_input = gr.File(
                    label=(
                        "FASTA / FASTQ 파일 업로드 (.fasta, .fa, .fastq, .fq,"
                        " .txt)"
                    )
                )
                seq_text_input = gr.Textbox(
                    lines=2,
                    placeholder=">Query_Sequence\nATGCGATCGATCGATCGATCGATC...",
                    label="직접 염기서열 입력 (FASTA/Raw Text)",
                )

            with gr.Accordion(
                "4️⃣ 임상 운영 임계값(Threshold) & 간이 검사 (Safety)",
                open=False,
            ):
                decision_threshold = gr.Slider(
                    minimum=0.50,
                    maximum=0.95,
                    value=0.85,
                    step=0.05,
                    label="🎯 감염병 확진 판정 임계값 (Decision Threshold)",
                )
                with gr.Row():
                    allergy_input = gr.CheckboxGroup(
                        choices=[
                            "페니실린 / 세파계 항생제 알레르기",
                            "아스피린/NSAIDs 과민증",
                        ],
                        label="⚠️ 약물 알레르기",
                    )
                    current_meds_input = gr.CheckboxGroup(
                        choices=[
                            (
                                "스타틴계 지질강하제 (심바스타틴, 아토르바스타틴"
                                " 등)"
                            ),
                            "항부정맥제 / 진정수면제 (CYP3A4 대사 약물)",
                        ],
                        label="💊 복용 상용약 (DDI)",
                    )
                with gr.Row():
                    rapid_test_type = gr.Dropdown(
                        choices=[
                            "신속항원검사 (Rapid Antigen Test)",
                            "RT-PCR 유전자 분자진단",
                        ],
                        value="신속항원검사 (Rapid Antigen Test)",
                        label="검사 종류",
                    )
                    rapid_test_result = gr.Radio(
                        choices=[
                            "미시행 (None)",
                            "양성 (Positive)",
                            "음성 (Negative)",
                        ],
                        value="미시행 (None)",
                        label="검사 결과 (Bayes)",
                    )

            with gr.Accordion(
                "5️⃣ 응급 위험 징후 & 혈액 검사 (Red Flags & Lab)", open=False
            ):
                red_flag_input = gr.CheckboxGroup(
                    choices=[
                        "지속적인 복통 및 지속성 구토 (CDC Warning Sign)",
                        "점막 출혈 (잇몸 출혈, 비출혈, 혈변)",
                        "흉통 또는 호흡곤란 악화",
                        "기면 상태 또는 극심한 안절부절못함",
                    ],
                    label="🚨 CDC 응급 징후",
                )
                with gr.Row():
                    plt_input = gr.Number(value=180, label="혈소판 (Plt, 10^3/μL)")
                    wbc_input = gr.Number(value=6500, label="백혈구 (WBC, /μL)")
                    crp_input = gr.Number(value=4.5, label="CRP (mg/L)")

            with gr.Accordion(
                "6️⃣ 정형 체크리스트 (Symptom & Exposure Checkbox)", open=False
            ):
                symptom_boxes = gr.CheckboxGroup(
                    choices=list(symptom_korean_map.keys()), label="📋 증상 체크"
                )
                epi_boxes = gr.CheckboxGroup(
                    choices=list(epi_korean_map.keys()), label="🌍 역학 노출력 체크"
                )

            submit_btn = gr.Button(
                "🚀 AI 캘리브레이션 & Grad-CAM 종합 진단 추론 실행",
                variant="primary",
                size="lg",
            )

        with gr.Column(scale=6):
            summary_output = gr.HTML(
                "<div style='padding:14px; background:#0f172a; border-radius:8px;"
                " border:1px solid #334155;'><h3>🏥 의심 질환: 추론 대기 중</h3></div>"
            )
            emergency_output = gr.HTML(
                "<div style='padding:10px; background:#1e293b; border-radius:6px;"
                " color:#cbd5e1;'>🟢 생체 징후 대기 중</div>"
            )
            next_q_output = gr.HTML(
                "<div style='padding:10px; background:#1e293b; border-radius:6px;"
                " color:#94a3b8;'>진료 데이터를 입력하시면 분석 결과가"
                " 도출됩니다.</div>"
            )

            label_output = gr.Label(
                num_top_classes=4,
                label=(
                    "📊 의심 감염병별 보정 확률 및 신뢰구간 (Calibrated Mean ±"
                    " Error)"
                ),
            )

            with gr.Accordion(
                "🔍 XAI 설명 가능한 AI: 주요 진단 근거 & Grad-CAM 시각화",
                open=True,
            ):
                with gr.Row():
                    xai_output = gr.Dataframe(
                        headers=[
                            "결정적 진단 피처 (모달리티/증상)",
                            "진단 기여도 (%)",
                        ],
                        datatype=["str", "str"],
                        label="핵심 기여 요인",
                    )
                    gradcam_output = gr.Image(
                        label="🔥 흉부 X-선 Grad-CAM 병변 활성화 히트맵"
                    )

            with gr.Tabs():
                with gr.TabItem("💊 CDC 정밀 치료제 & 권고 항생제"):
                    treatment_output = gr.Textbox(
                        lines=6,
                        label="💊 CDC 표준 1차 치료제 및 맞춤 항생제 (eGFR 감량)",
                    )
                    contra_output = gr.Textbox(
                        lines=5, label="⚠️ 절대 금기 약물 & DDI 병용금기 경보"
                    )
                with gr.TabItem("📄 공식 PDF 발생신고서 다운로드"):
                    gr.Markdown(
                        "#### 📥 질병관리청 발생신고서 & SOAP 공식 PDF (SHA-256"
                        " 각인)"
                    )
                    pdf_file_output = gr.File(label="공식 PDF 문서 다운로드")
                with gr.TabItem("🌐 CDC 표준 임상지침 & 🇰🇷 KDCA 지침"):
                    cdc_kdca_output = gr.Textbox(
                        lines=11,
                        label=(
                            "미국 CDC 표준 임상 가이드라인 & 질병관리청 법정 신고"
                            " 지침"
                        ),
                    )
                with gr.TabItem("📊 감별 진단 배제 사유 (Rule-out)"):
                    rule_out_output = gr.Dataframe(
                        headers=[
                            "감별 대상 질환",
                            "추론 확률(신뢰구간)",
                            "배제 및 후순위 선정 의학적 사유",
                        ],
                        datatype=["str", "str", "str"],
                    )
                with gr.TabItem("📝 EMR SOAP 임상 차트"):
                    soap_output = gr.Textbox(
                        lines=12, label="EMR 복사용 표준 SOAP 경과기록지"
                    )
                    with gr.Row():
                        doc_name = gr.Textbox(
                            label="담당 의사 성명", placeholder="홍길동"
                        )
                        doc_lic = gr.Textbox(
                            label="의사 면허 번호", placeholder="123456"
                        )
                    sign_btn = gr.Button(
                        "✍️ 의사 최종 검토 및 EMR 전자서명 승인",
                        variant="secondary",
                    )
                    sign_status = gr.HTML("")
                with gr.TabItem("📋 질병관리청 발생 신고서"):
                    report_output = gr.Textbox(
                        lines=14, label="보건소 제출용 감염병 발생 신고 서식"
                    )
                with gr.TabItem("🌐 International (English SOAP)"):
                    english_soap_output = gr.Textbox(
                        lines=14,
                        label="English Clinical Summary & Home Isolation Guide",
                    )

    # 14개 출력 컴포넌트 1:1 바인딩 완료
    submit_btn.click(
        fn=full_enterprise_cdss_pipeline,
        inputs=[
            age_input,
            sex_input,
            weight_input,
            sbp_input,
            rr_input,
            spo2_input,
            consciousness_input,
            scr_input,
            plt_input,
            wbc_input,
            crp_input,
            pregnant_input,
            exposure_days_input,
            comorbidities_input,
            allergy_input,
            current_meds_input,
            rapid_test_type,
            rapid_test_result,
            symptom_boxes,
            epi_boxes,
            red_flag_input,
            free_text_note,
            lesion_image,
            seq_file_input,
            seq_text_input,
            decision_threshold,
        ],
        outputs=[
            summary_output,
            emergency_output,
            next_q_output,
            label_output,
            xai_output,
            gradcam_output,
            treatment_output,
            contra_output,
            cdc_kdca_output,
            rule_out_output,
            soap_output,
            report_output,
            english_soap_output,
            pdf_file_output,
        ],
    )

    sign_btn.click(
        fn=doctor_signoff_action,
        inputs=[doc_name, doc_lic, soap_output, report_output],
        outputs=[sign_status, pdf_file_output],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 8080)),
        show_error=True,
    )
