"""python3 -m ai_layer prepare-index | demo"""

import argparse
import json
from pathlib import Path

from .config import ROOT
from .explainer import AIExplainer


def main():
    parser = argparse.ArgumentParser(description="AI-слой объяснений подрядчиков")
    parser.add_argument("command", choices=("prepare-index", "demo"))
    parser.add_argument("--case", help="id одного сценария из examples/demo-cases.json")
    args = parser.parse_args()
    explainer = AIExplainer()
    if args.command == "prepare-index":
        if not explainer.settings.openai_api_key:
            parser.error("заполните OPENAI_API_KEY в локальном .env")
        count = explainer.prepare_index()
        print("Индекс подготовлен: {} фрагментов. Ключ не сохранялся в индексе.".format(count))
        return
    path = ROOT / "examples/demo-cases.json"
    cases = json.loads(path.read_text(encoding="utf-8"))
    if args.case:
        cases = [case for case in cases if case["id"] == args.case]
        if not cases:
            parser.error("неизвестный id сценария")
    print("Демо только AI-слоя: id карточек взяты из проверенных сценариев, фильтрация backend здесь не запускается.")
    for case in cases:
        cards = []
        for profile_id in case["expected"]["card_ids"]:
            row = explainer.catalog.profiles[profile_id]
            cards.append({"id": profile_id, "anon_name": row["anon_name"],
                          "category": case["request"]["category"], "city": row["city"],
                          "price_from_kzt": int(row["price_from_kzt"]),
                          "synthetic": row["synthetic"] == "True",
                          "city_imputed": row["city_imputed"] == "True",
                          "price_imputed": row["price_imputed"] == "True"})
        response = {"status": case["expected"]["status"], "cards": cards}
        result = explainer.enrich_response(case["request"], response)
        print(json.dumps({"case": case["id"], "cards": result["cards"]},
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
