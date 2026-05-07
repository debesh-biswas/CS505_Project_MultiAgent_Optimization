# Natural Plan (NP) metrics adapted from google-deepmind/natural-plan (Apache-2.0).
# Trip: evaluate_trip_planning.py | Meeting: evaluate_meeting_planning.py | Calendar: evaluate_calendar_scheduling.py

from __future__ import annotations

import collections
import datetime
import json
import re
from typing import Any, List, Sequence, Tuple

# --- Trip planning (regex parse + exact match) ---


def parse_trip_response(response: str) -> List[Tuple[str, int]]:
    """Port of natural-plan ``parse_response``: list of (city, stay_days)."""
    pattern_visit = r"\d+-\d+"
    pattern_flight = r".*Day (\d+).*from (\w+) to (\w+)"
    pattern_days = r"European cities for (\d+) days"
    days: List[str] = []
    flights: List[Tuple[str, str, str]] = []
    flight_days: List[int] = []
    total_days = None
    for piece in response.split("\n"):
        days_match = re.findall(pattern_days, piece)
        if days_match:
            total_days = int(days_match[0])
        visit_match = re.findall(pattern_visit, piece)
        if visit_match:
            days.append(visit_match[0])
            end_day = int(visit_match[0].split("-")[1])
            if total_days is not None and end_day == total_days:
                break
        flight_match = re.findall(pattern_flight, piece)
        if flight_match:
            flights.append(flight_match[0])
    visit_cities: List[str] = []
    parsed_plan: List[Tuple[str, int]] = []
    for flight_day, begin_city, end_city in flights:
        flight_days.append(int(flight_day))
        if not visit_cities:
            visit_cities.append(begin_city)
            visit_cities.append(end_city)
        else:
            visit_cities.append(end_city)

    if not days or not flights or not visit_cities:
        return []
    last_day = int(days[-1].split("-")[1])
    flight_days = [1] + flight_days + [last_day]
    for i, visit_city in enumerate(visit_cities):
        city_stay = flight_days[i + 1] - flight_days[i] + 1
        parsed_plan.append((visit_city, city_stay))
    return parsed_plan


def trip_exact_match(cities: str, durations: str, model_response: str) -> float:
    """1.0 iff parsed plan exactly matches all city** stays and duration** days."""
    parsed_plan = parse_trip_response(model_response)
    stays = [x for x in cities.split("**") if x]
    day_list = [int(x) for x in durations.split("**") if x]
    if not stays or not parsed_plan:
        return 0.0
    num_stays = min(len(stays), len(parsed_plan))
    num_match = 0
    for i in range(num_stays):
        if stays[i] == parsed_plan[i][0] and day_list[i] == parsed_plan[i][1]:
            num_match += 1
        else:
            break
    return 0.0 if num_match / len(stays) < 1.0 else 1.0


# --- Meeting planning (text steps + validator score vs golden score) ---


def convert_to_time_obj(time_str: str) -> datetime.datetime:
    return datetime.datetime.strptime(time_str.strip(), "%I:%M%p")


def process_constraints(data: Sequence[Tuple[Any, ...]]):
    """Port of ``process_constraints`` (typo ``contraints`` kept as local name)."""
    contraints: dict = collections.defaultdict(dict)
    for name, location, times, meeting_time in data:
        contraints[name]["location"] = location
        start_time = convert_to_time_obj(times.split("to")[0].strip())
        end_time = convert_to_time_obj(times.split("to")[1].strip())
        contraints[name]["start_time"] = start_time
        contraints[name]["end_time"] = end_time
        contraints[name]["meeting_time"] = meeting_time
    return contraints


def validator_from_text(
    plan: List[str],
    processed_constraints: dict[str, Any],
    start_location: str,
    initial_time: str,
    dist_matrix: dict[str, Any],
) -> int:
    """Port of ``validator_from_text``: count of valid meetings scheduled."""
    met_with: dict = {}
    score = 0
    cur_location = start_location
    cur_time = convert_to_time_obj(initial_time)
    for step in plan:
        try:
            if step.startswith("You start"):
                continue
            if step.startswith("You travel"):
                destination = step.split("travel to ")[1].split(" in")[0].strip()
                cur_time = cur_time + datetime.timedelta(
                    minutes=dist_matrix[cur_location][destination]
                )
                cur_location = destination
            elif step.startswith("You wait"):
                raw_end_time = step.split("wait until ")[1].split(".")[0].strip()
                end_time = convert_to_time_obj(raw_end_time)
                if end_time <= cur_time:
                    raise ValueError("Cannot go backwards in time")
                cur_time = end_time
            elif step.startswith("You meet"):
                person = step.split("meet ")[1].split(" for")[0].strip()
                if person in met_with:
                    raise ValueError(
                        "Person {person} already met with {met_with}".format(
                            person=person, met_with=met_with[person]
                        )
                    )
                met_with[person] = 1
                new_time = cur_time + datetime.timedelta(
                    minutes=processed_constraints[person]["meeting_time"]
                )
                if (
                    cur_location == processed_constraints[person]["location"]
                    and cur_time >= processed_constraints[person]["start_time"]
                    and new_time <= processed_constraints[person]["end_time"]
                ):
                    score += 1
                    cur_time = new_time
                else:
                    raise ValueError("Invalid meeting time or location")
            else:
                raise ValueError("Unknown plan format")
        except (ValueError, KeyError, IndexError) as e:
            # Match upstream script: print and break (we only need final score).
            _ = e
            break
    return score


def parse_text_plan(plan: str) -> List[str]:
    """Port of ``parse_text_plan``."""
    prefix = "SOLUTION:"
    if prefix in plan:
        plan = plan[plan.find(prefix) + len(prefix):].strip()
    plan = plan.split(".")
    plan = [step.strip() for step in plan]
    final_plan = []
    for step in plan:
        if step:
            final_plan.append(step)
    return final_plan


def meeting_exact_match(
    pred_plan: str,
    golden_plan: str,
    constraints_rows: List[List[Any]],
    dist_matrix: dict[str, Any],
) -> float:
    """
    1.0 iff validator score on pred equals score on golden (Natural Plan meeting metric).
    ``constraints_rows`` is the decoded JSON list: row0 = [start_loc, start_time], rest = person rows.
    """
    start_location, initial_time = constraints_rows[0][0], constraints_rows[0][1]
    processed = process_constraints([tuple(r) for r in constraints_rows[1:]])
    pred_steps = parse_text_plan(pred_plan)
    golden_steps = parse_text_plan(golden_plan)
    score_p = validator_from_text(
        pred_steps, processed, start_location, initial_time, dist_matrix
    )
    score_g = validator_from_text(
        golden_steps, processed, start_location, initial_time, dist_matrix
    )
    return 1.0 if score_p == score_g else 0.0


# --- Calendar scheduling (regex slot + exact match) ---


def hour_to_num(hr_str: str) -> float:
    parts = hr_str.strip().split(":")
    h = float(parts[0])
    m = parts[1] if len(parts) > 1 else "00"
    return h + (0.5 if m == "30" else 0.0)


def parse_calendar_slot(response: str) -> Tuple[str, float, float]:
    """Port of ``_parse_response``: (day, start_hour, end_hour)."""
    time_strs = re.findall(
        r"[A-Za-z]+, [0-9]+:[0-9]+ - [0-9]+:[0-9]+", response
    )
    if not time_strs:
        return "", -1.0, -1.0
    time_str = time_strs[0]
    day, hour_str = (
        time_str.split(",")[0].strip(),
        time_str.split(",")[1].strip(),
    )
    start_hour, end_hour = (
        hour_str.split("-")[0].strip(),
        hour_str.split("-")[1].strip(),
    )
    return day, hour_to_num(start_hour), hour_to_num(end_hour)


def calendar_exact_match(model_response: str, golden_response: str) -> float:
    """1.0 iff parsed (day, start, end) tuples match."""
    r_day, r_s, r_e = parse_calendar_slot(model_response)
    g_day, g_s, g_e = parse_calendar_slot(golden_response)
    if r_day == g_day and r_s == g_s and r_e == g_e:
        return 1.0
    return 0.0


def load_constraints_and_dist(
    constraints_json: str, dist_matrix_json: str
) -> Tuple[List[List[Any]], dict[str, Any]]:
    rows = json.loads(constraints_json)
    dist_matrix = json.loads(dist_matrix_json)
    return rows, dist_matrix
