from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

def predict_region(s):
    try:
        lang = detect(s)
        region_map = {
            'en': 'us-en',
            # 'ca': 'ct-ca',
            'zh-cn': 'tw-tzh',
            'zh-tw': 'tw-tzh',
            # 'fr': 'fr-fr',
            # 'tr': 'tr-tr',
            # 'nl': 'nl-nl',
        }
        return region_map.get(lang, 'us-en') 
        
    except LangDetectException as e:
        print(f"Warning: Language detection failed: {e}. Using default region 'us-en'")
        return 'us-en'

def pwarn(str):
    print("\033[33m"+str+"\033[0m")