import os
import csv
import json
import time
from datetime import datetime

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

#Достаём настройки из .env
api_key = os.getenv("OPENROUTER_API_KEY")
model = os.getenv("MODEL", "cohere/north-mini-code:free")

base_url = "https://openrouter.ai/api/v1"

input_file = "data/news.csv"
output_file = "output/results.json"

#Системный промпт - объясняем модели её роль
system_prompt = "Ты - редактор новостной ленты. Делаешь короткие пересказы новостей. Отвечай строго в формате JSON, без пояснений и без markdown."

# Шаблон запроса, сюда подставляем заголовок и текст новости
user_prompt = """Сделай краткое содержание новости и верни JSON с полями:
- "summary": краткое содержание новости в 1-2 предложениях на русском
- "key_points": список из 2-3 ключевых фактов (массив коротких строк)
- "category": рубрика новости одним словом (например: технологии, экономика, спорт, наука, город, культура)

Заголовок: "{title}"
Текст новости: "{text}"
"""


def read_news(path):
    #Читаем новости из csv. Колонки: id, title, text
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "id": int(row["id"]),
                "title": row["title"].strip(),
                "text": row["text"].strip()
            })
    return rows


def extract_json(text):
    #Если модель присылает json внутри блока ```json ... ```, срезаем эту обёртку
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    return json.loads(text)


def summarize_news(client, title, text):
    #Отправляем новость в модель и забираем ответ
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(title=title, text=text)}
        ],
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    answer = response.choices[0].message.content
    return extract_json(answer)


def main():
    #Если ключа нет, то выходим
    if not api_key:
        raise SystemExit("Не задан OPENROUTER_API_KEY. Создай файл .env по образцу .env.example")

    client = OpenAI(api_key=api_key, base_url=base_url)

    news = read_news(input_file)
    print("Загружено новостей:", len(news))

    results = []
    for n in news:
        try:
            data = summarize_news(client, n["title"], n["text"])
            item = {
                "id": n["id"],
                "title": n["title"],
                "summary": data.get("summary"),
                "key_points": data.get("key_points"),
                "category": data.get("category")
            }
            print("[", n["id"], "]", data.get("category"), "- готово")
        except Exception as e:
            item = {"id": n["id"], "title": n["title"], "error": str(e)}
            print("[", n["id"], "] ошибка:", e)
        results.append(item)
        time.sleep(0.5)  # пауза чтобы не упереться в лимит запросов

    #Складываем всё в один словарь 
    output = {
        "task": "news_summarization",
        "model": model,
        "processed_at": datetime.now().isoformat(timespec="seconds"),
        "total": len(results),
        "results": results
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("Готово, результат сохранён в", output_file)


if __name__ == "__main__":
    main()
