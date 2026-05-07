"""
TravelPlanner tool wrappers for AutoGen function calling.

Agents with capability="WebSearch" get these tools registered so the LLM
can actually call search_flights, search_accommodations, etc. against the
real TravelPlannerDB CSV databases.
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))

_repo_candidates = [
    os.path.abspath(os.path.join(_here, "..", "..", "..", "scratch", "TravelPlanner_repo")),
    os.path.abspath(os.path.join(_here, "..", "..", "..", "TravelPlannerDB")),
    os.path.abspath(os.path.join(_here, "..", "..", "..", "..", "Project", "TravelPlannerDB")),
]
_repo_path = None
for _cand in _repo_candidates:
    if os.path.isfile(os.path.join(_cand, "tools", "flights", "apis.py")):
        _repo_path = _cand
        break

# Flag: tools available only if we found the repo and databases
_tools_available = False
_flights_api = None
_accommodations_api = None
_restaurants_api = None
_attractions_api = None
_distance_api = None

if _repo_path is not None:
    if _repo_path not in sys.path:
        sys.path.insert(0, _repo_path)
    _original_dir = os.getcwd()
    try:
        os.chdir(os.path.join(_repo_path, "tools"))
        from tools.flights.apis import Flights
        from tools.accommodations.apis import Accommodations
        from tools.restaurants.apis import Restaurants
        from tools.attractions.apis import Attractions
        from tools.googleDistanceMatrix.apis import GoogleDistanceMatrix

        _flights_api = Flights()
        _accommodations_api = Accommodations()
        _restaurants_api = Restaurants()
        _attractions_api = Attractions()
        _distance_api = GoogleDistanceMatrix()
        _tools_available = True
        print("[tp/tools] TravelPlanner DB APIs loaded successfully.")
    except Exception as _e:
        print(f"[tp/tools] Warning: could not load TravelPlanner APIs: {_e}")
    finally:
        os.chdir(_original_dir)
else:
    print("[tp/tools] Warning: TravelPlannerDB not found — WebSearch tools unavailable.")


def search_flights(origin: str, destination: str, departure_date: str) -> str:
    """Search for flights by origin city, destination city, and departure date (YYYY-MM-DD)."""
    if not _tools_available:
        return "Flight search unavailable (TravelPlannerDB not loaded)."
    try:
        results = _flights_api.run(origin, destination, departure_date)
        if isinstance(results, str):
            return results
        return results.to_json(orient="records") if len(results) > 0 else "No flights found."
    except Exception as e:
        return f"Error searching flights: {e}"


def search_accommodations(city: str) -> str:
    """Search for hotels/accommodations in a specific city."""
    if not _tools_available:
        return "Accommodation search unavailable (TravelPlannerDB not loaded)."
    try:
        results = _accommodations_api.run(city)
        if isinstance(results, str):
            return results
        return results.to_json(orient="records") if len(results) > 0 else "No accommodations found."
    except Exception as e:
        return f"Error searching accommodations: {e}"


def search_restaurants(city: str) -> str:
    """Search for restaurants in a specific city."""
    if not _tools_available:
        return "Restaurant search unavailable (TravelPlannerDB not loaded)."
    try:
        results = _restaurants_api.run(city)
        if isinstance(results, str):
            return results
        return results.to_json(orient="records") if len(results) > 0 else "No restaurants found."
    except Exception as e:
        return f"Error searching restaurants: {e}"


def search_attractions(city: str) -> str:
    """Search for tourist attractions in a specific city."""
    if not _tools_available:
        return "Attraction search unavailable (TravelPlannerDB not loaded)."
    try:
        results = _attractions_api.run(city)
        if isinstance(results, str):
            return results
        return results.to_json(orient="records") if len(results) > 0 else "No attractions found."
    except Exception as e:
        return f"Error searching attractions: {e}"


def get_distance_and_cost(origin: str, destination: str, mode: str) -> str:
    """
    Get distance and cost between two cities.
    mode: 'self-driving' or 'taxi'.
    """
    if not _tools_available:
        return "Distance/cost lookup unavailable (TravelPlannerDB not loaded)."
    try:
        results = _distance_api.run_for_evaluation(origin, destination, mode)
        if results["cost"] is None:
            return f"No route found for {mode} from {origin} to {destination}."
        return (
            f"Cost: ${results['cost']}, Distance: {results['distance']} miles, "
            f"Duration: {results['duration']} hours."
        )
    except Exception as e:
        return f"Error calculating distance: {e}"


WEBSEARCH_TOOLS = [
    search_flights,
    search_accommodations,
    search_restaurants,
    search_attractions,
    get_distance_and_cost,
]


def register_travel_tools(assistant_agent, user_proxy_agent):
    """
    Register all WebSearch tools on an AssistantAgent (for LLM schema) and
    on the UserProxyAgent (for execution). Call once per WebSearch agent.
    """
    for fn in WEBSEARCH_TOOLS:
        assistant_agent.register_for_llm(name=fn.__name__, description=fn.__doc__)(fn)
        user_proxy_agent.register_for_execution(name=fn.__name__)(fn)
    print(f"[tp/tools] Registered travel tools for {assistant_agent.name}")
