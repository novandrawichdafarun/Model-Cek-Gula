from fastapi import FastAPI, UploadFile, File, Form
from google import genai
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
import io, os
import time, random, asyncio
from typing import List, Optional

load_dotenv()
client = genai.Client()
app = FastAPI(title="Cek-Gula API", description="API untuk klasifikasi jajanan dan informasi nutrisi")

class NormalisasiGambar(tf.keras.layers.Layer):
  def call(self, inputs):
    inputs = tf.cast(inputs, tf.float32)
    return (inputs / 127.5) - 1.0

_path_model = os.getenv("BEST_MODEL") or os.getenv("BACKUP_MODEL")
_model = None

def get_model():
  global _model
  if _model is None:
    if not _path_model:
      raise RuntimeError("BEST_MODEL atau BACKUP_MODEL belum diset di .env")
    path = os.path.expanduser(_path_model)
    if not os.path.exists(path):
      raise FileNotFoundError(f"Model file not found: {path}")
    _model = tf.keras.models.load_model(path, custom_objects={'NormalisasiGambar': NormalisasiGambar})
  return _model

GENAI_MODEL = os.getenv("GEMINI_MODEL") or os.getenv("GENAI_MODEL") or "models/gemini-2.5-flash"

kelas_jajanan = [
  'ampyang', 'arem-arem', 'arum manis', 'bakpao', 'bakwan', 
  'biji salak', 'bika ambon', 'bolen', 'bolu gulung', 'bolu karamel', 
  'bolu kukus', 'brem', 'bubur mutiara', 'bubur sumsum', 'cenil', 
  'cilok', 'cimol', 'cireng', 'combro', 'dadar gulung', 'dodol', 
  'donat', 'gathot', 'geplak', 'getas', 'gethuk pisang', 'getuk goreng', 
  'getuk lindri', 'kembang goyang', 'keripik pisang', 'keripik singkong', 
  'keripik tempe', 'keukarah', 'klepon', 'kroket', 'kue akar kelapa', 
  'kue ape', 'kue apem', 'kue balok', 'kue bangkit', 'kue bawang', 
  'kue bingka', 'kue carabikang', 'kue cubit', 'kue cucur', 'kue gemblong', 
  'kue jipang', 'kue kacang', 'kue kamir', 'kue klemben', 'kue lekker', 
  'kue lidah kucing', 'kue lumpang', 'kue lumpur', 'kue lupis', 'kue manco', 
  'kue mendut', 'kue mochi', 'kue padamaran', 'kue pancong', 'kue pastel', 
  'kue pasung', 'kue perut ayam', 'kue pudak', 'kue pukis', 'kue putri mandi', 
  'kue putu', 'kue putu ayu', 'kue rangi', 'kue sagon', 'kue semprong', 'kue sus', 
  'kue talam', 'kue thok', 'kue wajik', 'kue yangko', 'lapis beras', 'lapis legit', 
  'lemper', 'lumpia', 'madu mongso', 'martabak manis', 'nagasari', 'onde-onde', 
  'otak-otak', 'panada', 'pisang goreng', 'rempeyek', 'rengginang', 'risol', 
  'sale pisang', 'sawut', 'serabi', 'sosis solo', 'tahu isi', 'tape ketan', 
  'tape singkong', 'tempe mendoan', 'tiwul', 'wingko babat'
]

try:
  df_nutrisi = pd.read_csv('nutrisi_jajanan.csv', index_col='nama_jajanan')
  NUTRISI = df_nutrisi.to_dict(orient='index')
except Exception:
  print("Warning: File nutrisi_jajanan.csv tidak ditemukan. Fitur nutrisi akan dinonaktifkan.")
  NUTRISI = {}

def kategori_resiko_gula(gylcemic_load):
  if gylcemic_load < 10:
    return "🟢 RENDAH", "Aman dikonsumsi, risiko lonjakan gula minimal"
  elif gylcemic_load < 20:
    return "🟡 SEDANG", "Konsumsi wajar, perhatikan porsi"
  else:
    return "🔴 TINGGI", "Batasi konsumsi, risiko sugar crash tinggi!"

class DataNutrisiInput(BaseModel):
  nama_jajanan: str
  gula_gram: float
  indeks_glikemik: int
  beban_glikemik: float
  status_risiko: str

@app.post("/prediksi/")
async def prediksi_gambar(file: UploadFile = File(...), porsi_gram: float = Form(100.0)):
  contents = await file.read()
  image = Image.open(io.BytesIO(contents)).convert('RGB')
  image = image.resize((224, 224))
  img_array = np.array(image)
  img_array = np.expand_dims(img_array, axis=0)
  
  model = get_model()
  prediksi = model.predict(img_array, verbose=0)
  indeks_tertinggi = np.argmax(prediksi[0])
  hasil_kelas = kelas_jajanan[indeks_tertinggi]
  keyakinan = float(prediksi[0][indeks_tertinggi])
  
  top3_idx = np.argsort(prediksi[0])[::-1][:3]
  top3_hasil = [{"jajanan": kelas_jajanan[i], "probabilitas": float(prediksi[0][i])} for i in top3_idx]
  
  data_nutrisi = None
  faktor_porsi = porsi_gram / 100.0
  
  if hasil_kelas in NUTRISI:
    n = NUTRISI[hasil_kelas]
    gl_hitung = n['GL'] * faktor_porsi
    risiko, pesan = kategori_resiko_gula(gl_hitung)
    
    data_nutrisi = {
      "kalori_kkal": round(n['kalori'] * faktor_porsi, 1),
      "karbohidrat_g": round(n['karbohidrat'] * faktor_porsi, 1),
      "gula_g": round(n['gula'] * faktor_porsi, 1),
      "protein_g": round(n['protein'] * faktor_porsi, 1),
      "lemak_g": round(n['lemak'] * faktor_porsi, 1),
      "indeks_glikemik": n['GI'],
      "beban_glikemik": round(gl_hitung, 1),
      "status_risiko": risiko,
      "pesan_kesehatan": pesan
    }
    
  return {
    "status": "success",
    "hasil_analisis": {
      "prediksi_utama": hasil_kelas,
      "keyakinan": keyakinan,
      "porsi_dihitung_gram": porsi_gram,
      "informasi_nutrisi": data_nutrisi,
      "prediksi_alternatif": top3_hasil
    }
  }

def _serialize_models(resp) -> List[str]:
    models = []
    try:
        for m in resp:
            if hasattr(m, "name"):
                models.append(m.name)
            elif isinstance(m, dict) and "name" in m:
                models.append(m["name"])
            else:
                models.append(str(m))
    except TypeError:
        try:
            if isinstance(resp, dict):
                for m in resp.get("models", []):
                    models.append(m.get("name") or str(m))
            elif isinstance(resp, list):
                for m in resp:
                    models.append(getattr(m, "name", getattr(m, "id", str(m))))
            else:
                models = [str(resp)]
        except Exception:
            models = [str(resp)]
    return models

def _get_available_models() -> List[str]:
    try:
        if hasattr(client, "list_models"):
            resp = client.list_models()
        elif hasattr(client, "models") and hasattr(client.models, "list"):
            resp = client.models.list()
        else:
            return []
        return _serialize_models(resp)
    except Exception:
        return []

def _normalize_model_name(name: str) -> str:
    if not name:
        return name
    return name if name.startswith("models/") else f"models/{name}"

def generate_with_retries(prompt: str, preferred: Optional[str] = None, per_model_retries: int = 2,
                          base_backoff: float = 1.0, backoff_factor: float = 2.0, max_models: int = 6):
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(_get_available_models())
    seen = set()
    candidates_norm = []
    for m in candidates:
        if not m:
            continue
        nm = _normalize_model_name(m)
        if nm not in seen:
            seen.add(nm)
            candidates_norm.append(nm)
    if not candidates_norm:
        candidates_norm = [_normalize_model_name(GENAI_MODEL)]
    last_exc = None
    for model_name in candidates_norm[:max_models]:
        for attempt in range(1, per_model_retries + 1):
            try:
                resp = client.models.generate_content(model=model_name, contents=prompt)
                text = getattr(resp, "text", None)
                if not text:
                    try:
                        text = resp.candidates[0].content
                    except Exception:
                        text = str(resp)
                return {"status": "success", "model": model_name, "saran_kesehatan": text}
            except Exception as e:
                last_exc = e
                msg = str(e).lower()
                if any(k in msg for k in ("unavailable", "503", "high demand", "rate limit")):
                    sleep = base_backoff * (backoff_factor ** (attempt - 1)) + random.random() * 0.5
                    time.sleep(sleep)
                    continue
                break
    return {"status": "error", "message": str(last_exc), "available_models": candidates_norm}

@app.post("/saran-kesehatan/")
async def berikan_saran(data: DataNutrisiInput):
    prompt = (
        f"Berperanlah sebagai ahli gizi profesional. Pengguna baru saja akan makan {data.nama_jajanan} "
        f"dengan kandungan gula {data.gula_gram} gram, Indeks Glikemik {data.indeks_glikemik}, "
        f"dan Beban Glikemik {data.beban_glikemik}. Status risikonya adalah {data.status_risiko}. "
        "Berikan saran edukatif dan praktis maksimal 3 kalimat tentang cara menyeimbangkan gula darah setelah memakan jajanan ini."
    )
    result = await asyncio.to_thread(generate_with_retries, prompt, GENAI_MODEL)
    return result

@app.get("/")
async def root():
  return {
    "message": "Selamat datang di Cek-Gula AI API",
    "dokumentasi": "Kunjungi http://127.0.0.1:8000/docs untuk mencoba API"
  }