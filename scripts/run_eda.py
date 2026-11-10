"""Build the EDA artifacts from an already collected local snapshot."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moex_portfolio.eda import build_report, execute_notebook  # noqa: E402


def write_notebook(root: Path) -> None:
    import nbformat

    summary = json.loads((root / "reports/eda/summary.json").read_text())
    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    cells = [
        nbformat.v4.new_markdown_cell(
            "# EDA: MOEX Portfolio Copilot\n\n"
            "Исследуется только development 2015–2024. Источники и хеши сырых ответов "
            "хранятся в `reports/eda/sources_manifest.json`; сырьё не публикуется. "
            "Квартальные составы индекса задают диагностическую панель. "
            "Этот ноутбук не является бэктестом и не оценивает прибыльность стратегии."
        ),
        nbformat.v4.new_code_cell(
            "import sys\nfrom pathlib import Path\n\nimport pandas as pd\n"
            "from IPython.display import Image, display\n"
            "ROOT = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent\n"
            "sys.path.insert(0, str(ROOT / 'src'))\n"
            "from moex_portfolio.eda import build_report  # noqa: E402\n"
            "summary = build_report(ROOT / 'data/processed', ROOT / 'reports/eda', "
            "config_path=ROOT / 'configs/eda.json')\n"
            "display(pd.DataFrame(summary['tables']).T)"
        ),
    ]
    sections = [
        (
            "Состав и качество рыночных данных",
            "coverage",
            "security_coverage",
            (
                f"В панели {summary['market']['securities']} тикеров и {summary['market']['rows']:,} свечей. "
                "Разная длина истории отражает и листинги/корпоративные события, и ограничения выбранного режима торгов. "
                "Отсутствие на дате MCFTR — диагностический флаг; его нельзя автоматически заполнять предыдущей ценой."
            ),
        ),
        (
            "Остановка торгов в 2022",
            "activity_2022",
            None,
            "Вокруг остановки меняются и число торгующихся инструментов, и оборот. "
            "Длинный календарный разрыв исключается из диагностических меток; в будущем журнале портфеля "
            "удерживаемые позиции и их фактический убыток через остановку должны сохраняться.",
        ),
        (
            "Диагностические таргеты 5 и 20 сессий",
            "price_targets",
            "target_summary",
            "Используются open(t+1+h)/open(t+1) и единый наблюдаемый календарь. "
            "На графике отдельно убраны интервалы с raw-скачком более 50%, распределение остальных значений "
            "обрезано только для отображения; сводная таблица сохраняет исходные экстремумы. "
            "Это доходности цен в том виде, в каком их ретроспективно отдаёт ISS; "
            "источник может менять масштаб истории. Поправки на корпоративные события и "
            "дивиденды независимо не проверены, total return не установлен.",
        ),
        (
            "Рыночные режимы",
            "regime_returns",
            "regime_summary",
            "Общий пул наблюдений меняется вместе с составом панели; различия распределений не являются "
            "статистическим доказательством изменения предсказуемости. Для оценки моделей нужны временные "
            "срезы и одинаковый point-in-time универсум.",
        ),
        (
            "Совместное движение бумаг",
            "correlation",
            "correlation_overlap",
            "Корреляция рассчитана без заполнения пропусков и при наличии минимум 252 общих наблюдений. "
            "Разные попарные окна не дают гарантию PSD; для портфельной ковариации потребуется общая матрица и shrinkage.",
        ),
        (
            "Новостные данные",
            "news_monthly",
            None,
            (
                f"Фактически загружено {summary['news']['rows']:,} строк за "
                f"{summary['news']['date_min']}–{summary['news']['date_max']}. "
                f"Повторных нормализованных текстов {summary['news']['normalized_text_duplicate_rows']:,}. "
                f"Время публикации не задано у {summary['news']['placeholder_counts']['time_not_stated']:,} строк. "
                "Такие статьи нельзя привязывать к закрытию торгов по выдуманному времени суток; "
                "нужна консервативная задержка либо исключение из строгой PIT-абляции. "
                f"Крупнейшая группа источник/день содержит {summary['news']['largest_source_day_groups'][0]['rows']:,} статей "
                f"({summary['news']['largest_source_day_groups'][0]['source']}, "
                f"{summary['news']['largest_source_day_groups'][0]['parsed_date']}); "
                "это аномалия, требующая проверки происхождения дат. "
                "Поставляемые train/test объединены только для описательной EDA; обучение потребует собственного "
                "временного разбиения с группировкой дубликатов. Покрытие новостями ограничивает период абляции."
            ),
        ),
        (
            "Покрытие компаний новостями",
            "news_coverage",
            "news_entity_coverage",
            "Применены регулярные выражения к заголовкам, текстам и tags после удаления точных нормализованных "
            "дубликатов. Это слабая привязка к компании: совпадение не означает тональность по конкретной бумаге. "
            "Знаменатель — пересечение диапазона новостей, истории компании и дат MCFTR; время суток пока не учтено.",
        ),
        (
            "Макроисточники",
            "macro",
            None,
            "Дата действия ставки/официального курса отличается от времени публикации. Предполагаемый backward as-of "
            "join должен использовать проверенный available_at; обратное заполнение и интерполяция из будущего запрещены.",
        ),
    ]
    for title, figure, table, conclusion in sections:
        cells.append(nbformat.v4.new_markdown_cell(f"## {title}\n\n{conclusion}"))
        source = (
            f"display(Image(filename=str(ROOT / 'reports/eda/figures/{figure}.png')))"
        )
        if table:
            source += (
                f"\ndisplay(pd.read_csv(ROOT / 'reports/eda/{table}.csv').head(15))"
            )
        cells.append(nbformat.v4.new_code_cell(source))
    cells.append(
        nbformat.v4.new_markdown_cell(
            "## Ограничения и следующие шаги\n\n"
            "1. Восстановить ежедневные интервалы состава индекса, корпоративные события и проверенный календарь.\n"
            "2. Проверить исторический момент доступности данных и часовой пояс новостей.\n"
            "3. Подготовить total-return метки и вложенную временную валидацию с purge по label_end_time.\n"
            "4. Сначала проверить ценовой baseline; сентимент сравнивать на том же периоде и универсуме.\n"
            "5. Holdout 2025–2026 открыть один раз после фиксации модели и правил исполнения."
        )
    )
    notebook.cells = cells
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for command in ["check", "format"]:
            arguments = [sys.executable, "-m", "ruff", command]
            if command == "check":
                arguments.extend(["--select", "I", "--fix"])
            arguments.extend(
                [
                    "--config",
                    str(root / "pyproject.toml"),
                    "--stdin-filename",
                    "notebook_cell.py",
                    "-",
                ]
            )
            result = subprocess.run(
                arguments, input=cell.source, capture_output=True, text=True, check=True
            )
            cell.source = result.stdout.rstrip()
    execute_notebook(notebook, root)
    destination = root / "notebooks/01_eda.ipynb"
    destination.parent.mkdir(exist_ok=True)
    nbformat.write(notebook, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use local processed files only (the default).",
    )
    parser.add_argument("--skip-notebook", action="store_true")
    args = parser.parse_args()
    summary = build_report(
        ROOT / "data/processed",
        ROOT / "reports/eda",
        config_path=ROOT / "configs/eda.json",
    )
    print(
        f"EDA complete: {summary['market']['rows']} candles, {summary['news']['rows']} news articles"
    )
    if not args.skip_notebook:
        write_notebook(ROOT)
        print("Executed notebook: notebooks/01_eda.ipynb")


if __name__ == "__main__":
    main()
