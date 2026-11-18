"""AI servisini başlatır. PyCharm'da bu dosyayı açıp yeşil ▶ butonuna bas.

Servis açılınca tarayıcıda http://127.0.0.1:8000/docs adresine git.
"""
import uvicorn

if __name__ == "__main__":
    # reload=True: kod değiştiğinde servis kendiliğinden yeniden başlar
    uvicorn.run("services.ai.main:app", host="127.0.0.1", port=8000, reload=True,
                reload_dirs=["services", "ml"])
