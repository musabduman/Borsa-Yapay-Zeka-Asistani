from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

class borsa_muhasebe:
    def __init__(self):
        self.model=RandomForestClassifier(n_estimators=200, min_samples_split=10,random_state=42,class_weight='balanced',max_depth=5)
    
    def analiz_et(self,df):
        if df is None or df.empty:
            return {"yön":"VERİ YOK","güven":"0"}
        
        df['Getiri']=df['Close'].pct_change()
        df['Hacim_degisimi']=df['Volume'].pct_change()
        df['Oynaklık']=(df['High']-df['Low'])/df['Close']
        
        delta=df['Close'].diff()
        
        gain=(delta.where(delta>0,0))
        lose=(-delta.where(delta<0,0))
        avg_gain=gain.ewm(com=13, adjust=False).mean()
        avg_lose=lose.ewm(com=13, adjust=False).mean()
        
        rs=avg_gain/avg_lose
        
        df['RSI']=100-(100/(1+rs))
        df['SMA_50']=df['Close']/df['Close'].rolling(window=50).mean()
        
        df['Momentum']=df['Close']/df['Close'].shift(10)
        
        df['RSI_Lag1']=df['RSI'].shift(1)
        df['Getiri_Lag1']=df['Getiri'].shift(1)
        df['Hacim_Lag1']=df['Hacim_degisimi'].shift(1)
        df['SMA_Uzaklik'] = (df['Close'] - df['Close'].rolling(window=50).mean()) / df['Close'].rolling(window=50).mean()
        df.replace([np.inf,-np.inf], np.nan, inplace=True)
        
        self.ozellikler=['Getiri','Hacim_degisimi','Oynaklık','RSI','SMA_50','Momentum',
                         'RSI_Lag1', 'Getiri_Lag1', 'Hacim_Lag1', 'SMA_Uzaklik']
        
        df['Target']=(df['Getiri'].shift(-1)>0.002).astype(int)
        
        bugun=df.iloc[[-1]][self.ozellikler]
        gecmis=df.dropna()
        
        X=gecmis[self.ozellikler]
        Y=gecmis['Target']

        spilt=int(len(X)*0.80)
        X_train,X_test=X.iloc[:spilt],X.iloc[spilt:]
        Y_train,Y_test=Y.iloc[:spilt],Y.iloc[spilt:]

        self.model.fit(X_train,Y_train)
        basari_puani=self.model.score(X_test,Y_test)
        print(f"Modelin başarı pauanı {basari_puani}")
        
        from sklearn.metrics import classification_report
        tahminler = self.model.predict(X_test)
        #print("Detaylı Karne:\n", classification_report(Y_test, tahminler))

        self.model.fit(X,Y)

        if bugun.isnull().values.any():
            return {"yön": "HESAPLANAMADI (Eksik Veri)", "güven": 0}

        tahmin=self.model.predict(bugun)[0]
        olasılık=self.model.predict_proba(bugun)[0]
        
        esik=0.55
        yukarı_yön=olasılık[1]
        asagi_yön=olasılık[0]
        if yukarı_yön>esik:
            return{"yön":"yükselis","güven":round(yukarı_yön*100, 2)}
        elif asagi_yön>esik:
            return{"yön":"düsüs","güven":round(asagi_yön*100, 2)}
        else:
            guven_puani = round(max(yukarı_yön,asagi_yön) * 100, 2)
            return {
                "yön": "NÖTR (Kararsız)", 
                "güven": guven_puani
            }
        
