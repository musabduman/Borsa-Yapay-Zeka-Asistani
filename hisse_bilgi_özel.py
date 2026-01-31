import yfinance as yf
import pandas as pd
from google import genai
import warnings
from datetime import datetime, timedelta
from ddgs import DDGS
import numpy as np
import time
from ilk_zeka import borsa_muhasebe
import ollama
import sys
import io

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

GOOGLE_API_KEY=" Buraya gemini apı keyinizi yazınız" 

client = genai.Client(api_key=GOOGLE_API_KEY)
pd.options.display.float_format = '{:.2f}'.format

def sembol_temizle(metin):
    tr_map = str.maketrans("igusocIGUSOC", "igusocIGUSOC")
    temiz_metin = metin.translate(tr_map).upper().strip()
    if not temiz_metin.endswith(".IS"):
        temiz_metin += ".IS"
    return temiz_metin

def teknik_analiz(df):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0))
    lose = (-delta.where(delta < 0, 0))
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_lose = lose.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_lose

    df['RSI'] = 100 - (100 / (1 + rs))
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['Volume_signal'] = volume_trend(df, window=60)
    df['Volatility'] = calcu_volatility(df, window=20)
    df = bollinger(df, window=20)
    df = calcu_macd(df)
    df = calcu_pivot(df)
    return df

def temel_veriler(hisse):
    info = hisse.info
    temel = {
        "FK Orani (P/E)": info.get('trailingPE', 'Veri Yok'),
        "PD/DD (P/B)": info.get('priceToBook', 'Veri Yok'),
        "Kar Marji (%)": info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 'Veri Yok',
        "Brut Kar": info.get('grossProfits', 'Veri Yok'),
        "Toplam Gelir": info.get('totalRevenue', 'Veri Yok'),
        "Hisse Basina Kar (EPS)": info.get('trailingEps', 'Veri Yok'),
        "Sektor": info.get('sector', 'Bilinmiyor'),
        "Oneri": info.get('recommendationKey', 'Yok')
    }
    return temel

def input_alma():
    while True:
        try:
            ham_girdi = input("Bilgi almak istediginiz hissenin ismini giriniz: ").upper()
            sembol = sembol_temizle(ham_girdi)
            hisse = yf.Ticker(sembol)
            df = hisse.history(period="1y")
            if df.empty:
                print("Veri bulunamadi.")
                return input_alma()
            return hisse, sembol, df
        except Exception as e:
            print(f"Baglanti hatasi: {e}")

def sinyal_kontrol(df):
    son = df.iloc[-1]
    wonderkid = (son['Width'] < 0.15) and (son['RSI'] < 60)
    erken_uyari = (son['MACD_signal'] == 1) and (son['Signal'] == 1)
    ralli = (son['MACD_signal'] == 1) and (son['Signal'] == 1) and (son['Volume_signal'] == 1)

    if ralli:
        return True, "Ralli modu"
    elif wonderkid:
        return True, "Wonderkid modu"
    elif erken_uyari:
        return True, "Erken uyari"
    return False, "Temiz"

def haber_verileri(sembol):
    haberler_listesi = []
    try:
        with DDGS() as ddgs:
            query = f"{sembol} hisse haberleri"
            result = ddgs.news(keywords=query, region="tr-tr", safesearch="off", max_results=5)
            for r in result:
                tarih = r.get('date', '')[:10]
                baslik = r.get('title', '')
                kaynak = r.get('source', '')
                haberler_listesi.append(f"-[{tarih}]{kaynak}:{baslik}")
    except:
        print("Haber verisi cekilemedi")
    return haberler_listesi

def bollinger(df, window):
    df['SMA'] = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=window).std()
    df['Upper'] = df['SMA'] + 2 * std
    df['Lower'] = df['SMA'] - 2 * std
    df['Width'] = (df['Upper'] - df['Lower']) / df['SMA']
    df['Signal'] = np.select(
        [df['Close'] > df['Upper'], df['Close'] < df['Lower']],
        [1, -1],
        default=0
    )
    return df

def volume_trend(df, window=10):
    df['volume_signal'] = np.where(
        df['Volume'] > df['Volume'].rolling(window=window).mean(), 1, 0
    )
    return df['volume_signal']

def calcu_volatility(df, window=20):
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=window).std()
    return df['Volatility']

def calcu_macd(df):
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_signal'] = np.where(df['MACD'] > df['Signal_line'], 1, -1)
    return df

def calcu_pivot(df):
    df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['R1'] = (2 * df['Pivot']) - df['Low']
    df['S1'] = (2 * df['Pivot']) - df['High']
    return df

def muhasebeci(hisse):
    df_muhasebeci = hisse.history(period="4y")
    try:
        bot = borsa_muhasebe()
        sonuc = bot.analiz_et(df_muhasebeci)
        return f"AI modeli %{sonuc['güven']} ihtimalle {sonuc['yon']} bekliyor."
    except Exception as e:
        return f"Hata: {e}"

def ollama_safe(text):
    if not isinstance(text, str):
        return text
    return text.encode("ascii", "ignore").decode()

def ollama_yorumla(temel, sembol, df, haberler_listesi, ai_rapor, analiz_sonucu, model="qwen3:4b"):
    son_veriler = df.tail(20).to_string()
    ai_rapor_safe = ollama_safe(ai_rapor)
    analiz_sonucu_safe = ollama_safe(analiz_sonucu)
    prompt = ollama_safe(f"""GÖREVİN: Sen dünyanın en iyi hedge fonlarında çalışan bir denetleyicisin sana gelen metini elindeki veriler ile denetle.
AMACIN:
Metini yeniden YAZMA. Sadece rapordaki mantıksal hatalar ve eksik verileri tespit et. 

1. TEKNİK_VERİLER:
{son_veriler}

2. AI_SKORU:
{ai_rapor_safe}

3. GEMINI_RAPORU (Bunu denetliyorsun):
{analiz_sonucu_safe}

KURALLAR:
- Gemini'nin edebi diline karışma.
- Sadece sayılar ve teknik indikatörler (RSI, MACD, Bollinger) dogru yorumlanmış mı ona bak.
- Eger Gemini "Yükseliş" demiş ama RSI 90 ise (aşırı pahalı), bunu uyarı olarak ekle.
- Eger Gemini önemli bir veriyi (örn: Hacim patlamasını) atlamışsa, onu ekle.

ÇIKTI FORMATI (Sadece aşagıdakini yaz):

[MANTIKÇI NOTLARI]
✅ ONAYLANANLAR:
⚠️ DÜZELTMELER:
➕ EKLENENLER:
DENETLE:
1. TEKNIK VERILER:
{son_veriler}

2. AI RAPOR:
{ai_rapor}

3. GEMINI RAPOR:
{analiz_sonucu}

CIKTI:
[MANTIKCI NOTLARI]
""")
    try:
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return f"{analiz_sonucu}\n{'='*40}\n{response['message']['content']}"
    except Exception as e:
        print(f"Ollama hatasi: {e}")
        return analiz_sonucu
def gemini_yorumla(temel, sembol, df, haberler_listesi, ai_rapor):
    son_veriler = df.tail(20).to_string()
    temel_metin = "\n".join([f"- {k}: {v}" for k, v in temel.items()])
    haberler_metni="\n".join(haberler_listesi) 
    son_veriler = df.tail(20).to_string()
    prompt = f"""Sen dünyanın en iyi hedge fonlarında çalışan bir borsa uzmanısın. 
    Sen karşındaki kişinin yatırım asistanısın; samimi, abartısız ve net bir dil kullanabilirsin (arkadaşça ama profesyonel). Sakın yatırım tavsiyesi verme sadece elindeki bilgileri yorumla !

    ÖNEMLİ: Yaptıgın son yorumda "Neden?" sorusuna cevap ver. Terimlere bogmadan, çokta uzatmadan, sonucun hangi veriden kaynaklandıgını açıkla. (Örn: "RSI 30'un altında oldugu için ucuz dedim" gibi).

    ELİNDEKİ VERİLER {sembol} İÇİN:

    1. TEMEL ANALİZ:
    {temel_metin}

    2. HABER AKIŞI (Son 1 Ay):
    {haberler_metni}
    (Haberlerin fiyat üzerindeki duygu durumunu -Sentiment- analiz et.)

    3. TEKNİK VERİLER (Son 20 Gün):
    {son_veriler}

    4. Aİ BOTU YARDIMI:
    {ai_rapor}
    (bu rapor tamamen sayısal verilerle hesaplanmıştır bunU AYNEN YAZDIR ve yorumunda kullan!)

    KARAR MEKANİZMAN (Bu kurallara sadık kal):
    • RSI: <30 (Aşırı Ucuz/Al Fırsatı), >70 (Aşırı Pahalı/Sat Fırsatı), 30-70 (Nötr/Trendi Takip Et).
    • MACD: 1 (Al/Yükseliş), -1 (Sat/Düşüş).
    • SMA 50/200: Fiyat ortalamanın üzerindeyse POZİTİF, altındaysa NEGATİF.
    • VOLUME_SIGNAL: 1 ise Yükseliş gerçek (Güven artır), 0 ise Yükseliş zayıf (Tuzak olabilir).
    • BOLLINGER: Width (Bant Genişligi) düşüyorsa "SIKIŞMA" var (Patlama Yakın). Signal 1 ise yukarı, 0 ise yatay.
    • PIVOT: Fiyat > Pivot ise Hedef R1. Fiyat < Pivot ise Destek S1.
    • VOLATİLİTE: Yüksekse stop seviyesini biraz daha geniş tut, düşükse dar tut.

    GÖREVİN:
    Tüm verileri (Temel + Teknik + Haber) birleştir. Teknik veriler "AL" derken Haberler "KÖTÜ" ise güven skorunu düşür. Çelişkileri belirt.

    ÇIKTI FORMATIN (Tam olarak bu başlıkları kullan):

    📊 GELECEK SENARYOSU:
    (İki üç cümle ile ne bekliyorsun? Yükseliş/Düşüş/Yatay)
    Karar mekanizmanda kullandıgın(MACD,SMA50,SMA200,VOLUME_SİGNAL,BOLLINGER,PİVOT,VOLATİLİTE,WİDTH) degerlerini burda satır satır göster ve yorumla !

    🎯 HEDEF FİYAT:
    (R1 veya teknik analize göre net bir rakam ver)

    🛑 STOP SEVİYESİ:
    (S1 veya risk yönetimine göre net bir rakam ver)

    🔥 GÜVEN SKORU:
    (0-100 arası. Neden bu puanı verdigini parantez içinde tek cümleyle açıkla.)

    📰 HABER VE TEMEL ETKİ:
    (Haberler teknigi destekliyor mu? Şirket temel olarak saglam mı?(kar marjını burda kullan) - En fazla 3 cümle)

    📈 TEKNİK ÖZET:
    (Göstergeler uyumlu mu? Hangi indikatör en baskın sinyali veriyor?)

    📌 SON KARAR:
    (GÜÇLÜ AL / AL / TUT / SAT / GÜÇLÜ SAT)
    VERILER:
    {son_veriler}

    AI RAPOR:
    {ai_rapor}
    """
    try:
        response = client.models.generate_content(
            model="models/gemini-flash-latest",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Gemini hatasi: {e}"
def main(): 
    soru=input("gemini detayli(tek)/ gemini detayli(bist30)/ sadece sayisal veri(tek)/ mega tarama sayisal(bist 100)?(1,2,3,4)")
    if soru=="1":    
        while True:
            hisse,sembol,df=input_alma()
            tr_map = str.maketrans("ığüşöçİĞÜŞÖÇ", "igusocIGUSOC")
            sembol = sembol.translate(tr_map).upper()
            try:
                df=teknik_analiz(df)
                temel=temel_veriler(hisse)
                ai_rapor=muhasebeci(hisse)
                haberler_listesi=haber_verileri(sembol)
                df.index = df.index.tz_localize(None)
                df_export = df.drop(["Dividends", "Stock Splits", "Volume"], axis=1, errors="ignore")
                df_export.to_excel(f"{sembol}_detayli_analiz.xlsx")

                analiz_sonucu=gemini_yorumla(temel,sembol,df,haberler_listesi,ai_rapor)
                final_rapor=ollama_yorumla(temel,sembol,df,haberler_listesi,ai_rapor,analiz_sonucu,model="qwen3:4b")
                print("="*60)
                print(final_rapor)
                print("="*60)
                print(ai_rapor)
                while True:
                    devam=input("Başka bir hisse sormak istiyor musunuz ? (E/H)").upper()
                    if devam=='E':
                        break
                    elif devam=='H':
                        print("İyi günler ")
                        return
                    else: 
                        print("Lütfen sadece H veya E giriniz.")
                
            except Exception as e:
                print(f"Beklenmeyen hata: {str(e)}")
    elif soru=="2":
        print("Tarama başlıyor...")
        firsat_listesi=[]
        bist30=[ "AKBNK.IS", "ALARK.IS", "ASELS.IS", "ASTOR.IS", "BIMAS.IS", "BRSAN.IS","CIMSA.IS",
                "DOAS.IS", "EKGYO.IS", "ENKAI.IS", "EREGL.IS", "FROTO.IS", "GARAN.IS","GUBRF.IS", "ULKER.IS", "ISCTR.IS", "KCHOL.IS", "KONTR.IS", "KOZAL.IS",
                "KRDMD.IS", "ODAS.IS", "OYAKC.IS", "PETKM.IS", "PGSUS.IS", "SAHOL.IS","SASA.IS", "SISE.IS", "TCELL.IS", "THYAO.IS", "TOASO.IS", "TUPRS.IS", "YKBNK.IS",
                "SMRTG.IS"]
        for sembol in bist30:
            try:
                hisse=yf.Ticker(sembol)
                df=hisse.history(period="1y")
                ai_rapor=muhasebeci(hisse)
                if df.empty: continue
                df=teknik_analiz(df)
                durum, sinyal=sinyal_kontrol(df)
                if durum:
                    print(f"Fırsat tesbit edildi {sembol} listeye ekleniyor...")
                    firsat_listesi.append((sembol,hisse,df))
                else:
                    print(f"{sembol} bu hisseden bir şey çikmaz")    

            except Exception as e:
                print(f"Hata: {e}")
                continue
        
        if len(firsat_listesi)>0:
            print(f"{len(firsat_listesi)} adet hisse tesbit edilmiştir detayli analiz başliyor...")
            for sembol,hisse,df in firsat_listesi:
                print(f"{sembol} analiz ediliyor...")
                
                temel=temel_veriler(hisse)
                haberler_listesi=haber_verileri(sembol)
                analiz_sonucu=gemini_yorumla(temel,sembol,df,haberler_listesi,ai_rapor)
                final_rapor=ollama_yorumla(temel,sembol,df,haberler_listesi,ai_rapor,analiz_sonucu,model="qwen3:4b")
                ai_rapor=muhasebeci(df)
                print(50*'*')
                print(final_rapor)
                print(50*'*')
                time.sleep(25)

        else:
            print("Bu bist30 listesinde akitf yükseliş trendi bulunan hisse bulunamadı:.()")

    elif soru=="3":
        hisse,sembol,df=input_alma()
        tr_map = str.maketrans("ığüşöçİĞÜŞÖÇ", "igusocIGUSOC")
        sembol = sembol.translate(tr_map).upper()
        ai_rapor1=muhasebeci(hisse)
        print(ai_rapor1)
    elif soru=="4":
        print("\n🚀 MEGA TARAMA MODU BAŞLATILIYOR (Sadece Yerel Yapay Zeka)")
        print("Google API kullanılmayacak, hız kesmek yok!\n")
        
        # BIST 100'den seçmece sağlam liste (İstediğini ekle/çıkar)
        bist100_listesi = [
            "AEFES.IS", "AGHOL.IS", "AKBNK.IS", "AKCNS.IS", "AKSA.IS", "AKSEN.IS", "ALARK.IS", "ALBRK.IS", "ALGYO.IS", "ALKIM.IS",
            "ARCLK.IS", "ASELS.IS", "ASTOR.IS", "BERA.IS", "BIMAS.IS", "BRSAN.IS", "BRYAT.IS", "BUCIM.IS", "CANTE.IS", "CCOLA.IS",
            "CEMTS.IS", "CIMSA.IS", "DOAS.IS", "DOHOL.IS", "ECILC.IS", "EGEEN.IS", "EKGYO.IS", "ENJSA.IS", "ENKAI.IS", "EREGL.IS",
            "EUREN.IS", "FROTO.IS", "GARAN.IS", "GENIL.IS", "GESAN.IS", "GLYHO.IS", "GUBRF.IS", "HALKB.IS", "HEKTS.IS", "IPEKE.IS",
            "ISCTR.IS", "ISDMR.IS", "ISGYO.IS", "ISMEN.IS", "IZMDC.IS", "KARSN.IS", "KCAER.IS", "KCHOL.IS", "KONTR.IS", "KORDS.IS",
            "KOZAL.IS", "KOZAA.IS", "KRDMD.IS", "MGROS.IS", "ODAS.IS", "OTKAR.IS", "OYAKC.IS", "PETKM.IS", "PGSUS.IS", "SAHOL.IS",
            "SASA.IS", "SISE.IS", "SKBNK.IS", "SMRTG.IS", "SNGYO.IS", "SOKM.IS", "TAVHL.IS", "TCELL.IS", "THYAO.IS", "TKFEN.IS",
            "TOASO.IS", "TSKB.IS", "TTKOM.IS", "TTRAK.IS", "TUKAS.IS", "TUPRS.IS", "ULKER.IS", "VAKBN.IS", "VESBE.IS", "VESTL.IS",
            "YKBNK.IS", "YYLGD.IS", "ZOREN.IS"
        ]

        yukselis_beklenenler = []

        for sembol in bist100_listesi:
            try:
                print(f"Borsa: {sembol} verisi çekiliyor...", end="\r") 
                hisse = yf.Ticker(sembol)
                df = hisse.history(period="1y")
                
                if df.empty: continue
                try:
                    bot = borsa_muhasebe() 
                    sonuc = bot.analiz_et(df) 
                    
                    yazı_rengi = "🚀" if "YÜKSELİŞ" in sonuc['yön'] else "🔻"
                    print(f"[{sembol}] -> %{sonuc['güven']} {sonuc['yön']} {yazı_rengi}")
                
                    if "YÜKSELİŞ" in sonuc['yön'] and sonuc['güven'] > 60:
                        yukselis_beklenenler.append((sembol, sonuc['güven']))

                except Exception as e_bot:
                    print(f"[{sembol}] Analiz Hatası: {e_bot}")
                time.sleep(0.5)

            except Exception as e:
                print(f"Hata ({sembol}): {e}")
                continue
        
        print("\n" + "="*40)
        print(f"🏆 TARAMA BİTTİ! OGLUNUN SEÇTİKLERİ ({len(yukselis_beklenenler)} Adet)")
        print("="*40)
        
        yukselis_beklenenler.sort(key=lambda x: x[1], reverse=True)
        
        for hisse, güven in yukselis_beklenenler:
            print(f"⭐ {hisse} - Güven: %{güven}")
        print("="*40 + "\n")
    else:
        print("Lütfen sadece 1 veya 2 degerini giriniz!!!")

    if input("Devam etmek istiyorsanız enter, bitirmek istiyorsanız q ya basınız.").lower()=='q':
        return
    else:
        main()

if __name__=="__main__":
    main()