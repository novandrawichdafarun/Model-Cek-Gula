from fastapi import FastAPI, UploadFile, File, Form
from google import genai
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
import io, os

load_dotenv()
client = genai.Client()
app = FastAPI(title="Cek-Gula API", description="API untuk klasifikasi jajanan dan informasi nutrisi")

class NormalisasiGambar(tf.keras.layers.Layer):
  def call(self, inputs):
    inputs = tf.cast(inputs, tf.float32)
    return (inputs / 127.5) - 1.0

model = tf.keras.models.load_model('./best_model.keras', custom_objects={'NormalisasiGambar': NormalisasiGambar})

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
except:
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
  status_resiko: str

@app.post("/prediksi/")
async def prediksi_gambar(file: UploadFile = File(...), porsi_gram: float = Form(100.0)):
  contents = await file.read()
  image = Image.open(io.BytesIO(contents)).convert('RGB')
  image = image.resize((224, 224))
  img_array = np.array(image)
  img_array = np.expand_dims(img_array, axis=0)
  
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
  
@app.post("/saran-kesehatan/")
async def berikan_saran(data: DataNutrisiInput):
  prompt = f"Berperanlah sebagai ahli gizi profesional. Pengguna baru saja akan makan {data.nama_jajanan} dengan kandungan gula {data.gula_gram} gram, Indeks Glikemik {data.indeks_glikemik}, dan Beban Glikemik {data.beban_glikemik}. Status risikonya adalah {data.status_risiko}. Berikan saran edukatif dan praktis maksimal 3 kalimat tentang cara menyeimbangkan gula darah setelah memakan jajanan ini."
  
  try:
    respons = client.models.generate_content(
      model='gemini-1.5-flash',
      contents=prompt
    )
    return {
      "status": "success",
      "saran_kesehatan": respons.text
    }
  except Exception as e:
    return {
      "status": "error",
      "message": str(e)
    }

@app.get("/")
async def root():
  return {
    "message": "Selamat datang di GulaGuard AI API", 
    "dokumentasi": "Kunjungi http://127.0.0.1:8000/docs untuk mencoba API"
  }