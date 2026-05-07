import os
import sys
from .logger import get_logger

logger = get_logger()

# Ensure TravelPlanner_repo is in path and we set the working directory 
# so the tools can find their databases with the default "../database/..." paths.
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scratch", "TravelPlanner_repo"))
if repo_path not in sys.path:
    sys.path.append(repo_path)

# Change directory temporarily to load the APIs correctly
original_dir = os.getcwd()
try:
    os.chdir(os.path.join(repo_path, "tools"))
    
    from tools.flights.apis import Flights
    from tools.accommodations.apis import Accommodations
    from tools.restaurants.apis import Restaurants
    from tools.attractions.apis import Attractions
    from tools.googleDistanceMatrix.apis import GoogleDistanceMatrix
    
    # Instantiate the tool backends
    _flights_api = Flights()
    _accommodations_api = Accommodations()
    _restaurants_api = Restaurants()
    _attractions_api = Attractions()
    _distance_api = GoogleDistanceMatrix()
finally:
    os.chdir(original_dir)


# AutoGen wrapper functions
def search_flights(origin: str, destination: str, departure_date: str) -> str:
    """Search for flights by origin, destination, and departure date (YYYY-MM-DD)."""
    try:
        results = _flights_api.run(origin, destination, departure_date)
        if isinstance(results, str): return results # Error message
        return results.to_json(orient="records") if len(results) > 0 else "No flights found."
    except Exception as e:
        logger.error(f"Error searching flights: {e}")
        return f"Error searching flights: {e}"

def search_accommodations(city: str) -> str:
    """Search for accommodations (hotels) in a specific city."""
    try:
        results = _accommodations_api.run(city)
        if isinstance(results, str): return results
        return results.to_json(orient="records") if len(results) > 0 else "No accommodations found."
    except Exception as e:
        logger.error(f"Error searching accommodations: {e}")
        return f"Error searching accommodations: {e}"

def search_restaurants(city: str) -> str:
    """Search for restaurants in a specific city."""
    try:
        results = _restaurants_api.run(city)
        if isinstance(results, str): return results
        return results.to_json(orient="records") if len(results) > 0 else "No restaurants found."
    except Exception as e:
        logger.error(f"Error searching restaurants: {e}")
        return f"Error searching restaurants: {e}"

def search_attractions(city: str) -> str:
    """Search for tourist attractions in a specific city."""
    try:
        results = _attractions_api.run(city)
        if isinstance(results, str): return results
        return results.to_json(orient="records") if len(results) > 0 else "No attractions found."
    except Exception as e:
        logger.error(f"Error searching attractions: {e}")
        return f"Error searching attractions: {e}"

def get_distance_and_cost(origin: str, destination: str, mode: str) -> str:
    """
    Get the distance and estimated cost between two cities.
    mode: Can be 'self-driving' or 'taxi'.
    """
    try:
        results = _distance_api.run_for_evaluation(origin, destination, mode)
        if results['cost'] is None:
            return f"No route found for {mode} from {origin} to {destination}."
        return f"Cost: ${results['cost']}, Distance: {results['distance']} miles, Duration: {results['duration']} hours."
    except Exception as e:
        logger.error(f"Error calculating distance: {e}")
        return f"Error calculating distance: {e}"

# Map for the GA framework
CAPABILITY_TO_TOOL = {
    "WebSearch": [search_flights, search_accommodations, search_restaurants, search_attractions, get_distance_and_cost],
}

def register_travel_tools(assistant, user_proxy):
    """Registers all travel tools to the assistant and user_proxy for AutoGen."""
    for tool_func in CAPABILITY_TO_TOOL["WebSearch"]:
        assistant.register_for_llm(name=tool_func.__name__, description=tool_func.__doc__)(tool_func)
        user_proxy.register_for_execution(name=tool_func.__name__)(tool_func)
    logger.info(f"Registered travel tools for {assistant.name}")
