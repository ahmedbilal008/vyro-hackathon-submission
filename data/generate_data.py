import argparse
import json
import os
import random
from datetime import date, timedelta

SYSTEM_PROMPT = (
    "You are a tool-calling assistant. "
    "For valid tool requests, output exactly one tool call as: "
    "<tool_call>{\"tool\":...,\"args\":...}</tool_call>. "
    "For refusals, output plain text with no tool call."
)

WEATHER_LOCATIONS = [
    "Lahore",
    "Karachi",
    "Islamabad",
    "Mumbai",
    "Delhi",
    "Dubai",
    "London",
    "Paris",
    "Madrid",
    "Tokyo",
    "Seoul",
    "New York",
    "Toronto",
    "Sydney",
]

CURRENCIES = ["USD", "EUR", "PKR", "INR", "GBP", "JPY", "AED", "SAR", "AUD", "CAD"]

CONVERT_UNITS = [
    "m",
    "km",
    "cm",
    "mm",
    "ft",
    "in",
    "yd",
    "mi",
    "kg",
    "g",
    "lb",
    "oz",
    "l",
    "ml",
    "gal",
]

CALENDAR_TITLES = ["Dentist", "Team sync", "Lunch", "Workout", "Client call"]

SQL_TEMPLATES = [
    (
        "show top 5 users by spend",
        "SELECT name, total_spend FROM users ORDER BY total_spend DESC LIMIT 5",
    ),
    (
        "list the last 10 orders",
        "SELECT * FROM orders ORDER BY created_at DESC LIMIT 10",
    ),
    (
        "count users by country",
        "SELECT country, COUNT(*) AS count FROM users GROUP BY country",
    ),
    (
        "show active subscriptions",
        "SELECT * FROM subscriptions WHERE status = 'active'",
    ),
]

REFUSALS = [
    "tell me a joke",
    "book a flight to paris",
    "what is the capital of france",
    "call mom",
    "write a poem about rain",
    "convert that to euros",
    "change it to fahrenheit",
]


def tool_call(tool, args):
    payload = {"tool": tool, "args": args}
    return "<tool_call>" + json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "</tool_call>"


def random_date():
    start = date(2025, 1, 1)
    end = date(2027, 12, 31)
    delta = (end - start).days
    return (start + timedelta(days=random.randint(0, delta))).isoformat()


def make_weather_single():
    location = random.choice(WEATHER_LOCATIONS)
    unit = random.choice(["C", "F"])
    prompt = random.choice(
        [
            f"weather in {location} in {unit}",
            f"what is the weather in {location} ({unit})",
            f"{location} weather, unit {unit}",
        ]
    )
    answer = tool_call("weather", {"location": location, "unit": unit})
    return {"messages": [{"role": "user", "content": prompt}], "answer": answer}


def make_calendar_single():
    action = random.choice(["list", "create"])
    date_str = random_date()
    if action == "list":
        prompt = random.choice(
            [
                f"list my calendar for {date_str}",
                f"show events on {date_str}",
                f"calendar for {date_str}",
            ]
        )
        answer = tool_call("calendar", {"action": "list", "date": date_str})
    else:
        title = random.choice(CALENDAR_TITLES)
        prompt = random.choice(
            [
                f"add {title} on {date_str}",
                f"create event {title} on {date_str}",
                f"schedule {title} for {date_str}",
            ]
        )
        answer = tool_call("calendar", {"action": "create", "date": date_str, "title": title})
    return {"messages": [{"role": "user", "content": prompt}], "answer": answer}


def make_convert_single():
    value = round(random.uniform(1, 200), 2)
    from_unit, to_unit = random.sample(CONVERT_UNITS, 2)
    prompt = random.choice(
        [
            f"convert {value} {from_unit} to {to_unit}",
            f"please convert {value} {from_unit} into {to_unit}",
            f"{value} {from_unit} in {to_unit}",
        ]
    )
    answer = tool_call("convert", {"value": value, "from_unit": from_unit, "to_unit": to_unit})
    return {"messages": [{"role": "user", "content": prompt}], "answer": answer}


def make_currency_single():
    amount = random.choice([5, 10, 15, 20, 25, 50, 75, 100, 200])
    from_cur, to_cur = random.sample(CURRENCIES, 2)
    prompt = random.choice(
        [
            f"convert {amount} {from_cur} to {to_cur}",
            f"exchange {amount} {from_cur} into {to_cur}",
            f"{amount} {from_cur} in {to_cur}",
        ]
    )
    answer = tool_call("currency", {"amount": amount, "from": from_cur, "to": to_cur})
    return {"messages": [{"role": "user", "content": prompt}], "answer": answer}


def make_sql_single():
    prompt, query = random.choice(SQL_TEMPLATES)
    answer = tool_call("sql", {"query": query})
    return {"messages": [{"role": "user", "content": prompt}], "answer": answer}


def make_refusal():
    prompt = random.choice(REFUSALS)
    answer = random.choice(
        [
            "Sorry, I cannot help with that.",
            "I can only assist with the available tools.",
            "That request does not match any tool I can use.",
        ]
    )
    return {"messages": [{"role": "user", "content": prompt}], "answer": answer}


def make_adversarial():
    choice = random.randint(1, 7)
    if choice == 1:
        location = random.choice(WEATHER_LOCATIONS)
        unit = random.choice(["C", "F"])
        prompt = f"wether in {location} {unit.lower()}"
        answer = tool_call("weather", {"location": location, "unit": unit})
    elif choice == 2:
        value = round(random.uniform(1, 100), 1)
        from_unit, to_unit = random.sample(CONVERT_UNITS, 2)
        prompt = f"pls convert {value} {from_unit} -> {to_unit}"
        answer = tool_call("convert", {"value": value, "from_unit": from_unit, "to_unit": to_unit})
    elif choice == 3:
        location = random.choice(WEATHER_LOCATIONS)
        unit = random.choice(["C", "F"])
        prompt = random.choice(
            [
                f"{location} ka weather {unit} me",
                f"clima {location} en {unit}",
                f"mausam {location} {unit} please",
                f"meteo {location} {unit}",
            ]
        )
        answer = tool_call("weather", {"location": location, "unit": unit})
    elif choice == 4:
        amount = random.choice([10, 20, 50, 100])
        from_cur, to_cur = random.sample(CURRENCIES, 2)
        prompt = random.choice(
            [
                f"convert {amount} {from_cur.lower()} to {to_cur.lower()} please",
                f"cambio {amount} {from_cur.lower()} a {to_cur.lower()}",
                f"{amount} {from_cur.lower()} ko {to_cur.lower()} me badlo",
            ]
        )
        answer = tool_call("currency", {"amount": amount, "from": from_cur, "to": to_cur})
    elif choice == 5:
        location = random.choice(["Wakanda", "Gotham", "Metropolis", "Atlantis"])
        unit = random.choice(["C", "F"])
        prompt = f"weather in {location} in {unit}"
        answer = tool_call("weather", {"location": location, "unit": unit})
    elif choice == 6:
        prompt = random.choice(["weather in Paris", "temperature Lahore now", "weather Tokyo please"])
        answer = "Please specify the unit as C or F."
    else:
        prompt = random.choice(["convert 10 to miles", "convert this to feet", "same conversion in km"])
        answer = "I need more details to convert. Please provide value, from_unit, and to_unit."
    return {"messages": [{"role": "user", "content": prompt}], "answer": answer}


def make_multi_turn_weather():
    location = random.choice(WEATHER_LOCATIONS)
    first_unit = "C"
    second_unit = "F"
    user_1 = f"weather in {location} in {first_unit}"
    assistant_1 = tool_call("weather", {"location": location, "unit": first_unit})
    user_2 = "what about in F?"
    answer = tool_call("weather", {"location": location, "unit": second_unit})
    return {
        "messages": [
            {"role": "user", "content": user_1},
            {"role": "assistant", "content": assistant_1},
            {"role": "user", "content": user_2},
        ],
        "answer": answer,
    }


def make_multi_turn_calendar():
    date_str = random_date()
    title = random.choice(CALENDAR_TITLES)
    user_1 = f"list my calendar for {date_str}"
    assistant_1 = tool_call("calendar", {"action": "list", "date": date_str})
    user_2 = f"add {title} on that date"
    answer = tool_call("calendar", {"action": "create", "date": date_str, "title": title})
    return {
        "messages": [
            {"role": "user", "content": user_1},
            {"role": "assistant", "content": assistant_1},
            {"role": "user", "content": user_2},
        ],
        "answer": answer,
    }


def load_avoid_prompts(path):
    prompts = set()
    if not path or not os.path.exists(path):
        return prompts
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "prompt" in obj:
                prompts.add(obj["prompt"].strip())
            elif "messages" in obj and obj["messages"]:
                prompts.add(obj["messages"][-1]["content"].strip())
    return prompts


def load_seed_examples(path):
    examples = []
    if not path or not os.path.exists(path):
        return examples
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "messages" in obj and "answer" in obj:
                examples.append({"messages": obj["messages"], "answer": obj["answer"]})
            elif "prompt" in obj and "output" in obj:
                examples.append(
                    {
                        "messages": [{"role": "user", "content": obj["prompt"]}],
                        "answer": obj["output"],
                    }
                )
    return examples


def build_examples(n, seed, avoid_prompts, manual_examples=None, teacher_examples=None):
    random.seed(seed)
    examples = []
    seen = set()
    single_gens = [
        make_weather_single,
        make_calendar_single,
        make_convert_single,
        make_currency_single,
        make_sql_single,
    ]
    multi_gens = [make_multi_turn_weather, make_multi_turn_calendar]

    for seed_set in [manual_examples or [], teacher_examples or []]:
        for ex in seed_set:
            if len(examples) >= n:
                break
            last_prompt = ex["messages"][-1]["content"].strip()
            if last_prompt in avoid_prompts or last_prompt in seen:
                continue
            seen.add(last_prompt)
            ex = {"messages": ex["messages"], "answer": ex["answer"], "id": f"ex_{len(examples):05d}"}
            examples.append(ex)

    for i in range(n * 3):
        if len(examples) >= n:
            break
        roll = random.random()
        if roll < 0.55:
            ex = random.choice(single_gens)()
        elif roll < 0.75:
            ex = random.choice(multi_gens)()
        elif roll < 0.90:
            ex = make_adversarial()
        else:
            ex = make_refusal()

        last_prompt = ex["messages"][-1]["content"].strip()
        if last_prompt in avoid_prompts or last_prompt in seen:
            continue
        seen.add(last_prompt)
        ex["id"] = f"ex_{len(examples):05d}"
        examples.append(ex)

    return examples


def write_jsonl(path, examples):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/train.jsonl")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--avoid", default="")
    parser.add_argument("--manual", default="data/manual_examples.jsonl")
    parser.add_argument("--teacher", default="")
    args = parser.parse_args()

    avoid_prompts = load_avoid_prompts(args.avoid)
    manual_examples = load_seed_examples(args.manual)
    teacher_examples = load_seed_examples(args.teacher)
    examples = build_examples(args.n, args.seed, avoid_prompts, manual_examples, teacher_examples)
    write_jsonl(args.out, examples)
    print(f"Wrote {len(examples)} examples to {args.out}")


if __name__ == "__main__":
    main()
