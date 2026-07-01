# ================================================================
# CutiGo: Flask API — app.py
# Deploy to Render (render.com)
#
# Endpoints:
#   GET  /              → health check
#   POST /recommend     → main itinerary endpoint
#   GET  /states        → list all available states
#   GET  /activities    → list all activity interests
#   GET  /places        → search places by name + state (MySQL)
#   GET  /famous-places → top is_famous places (MySQL)
#   POST /trips         → save a trip (MySQL)
#   GET  /trips         → list a user's saved trips (MySQL)
#   PUT  /trips/<id>    → edit a saved trip (MySQL)
#   DELETE /trips/<id>  → delete a saved trip (MySQL)
#
# Folder structure on Render:
#   /
#   ├── app.py
#   ├── requirements.txt
#   ├── ml_model/
#   │   ├── cutigo_rf_model.pkl
#   │   ├── cutigo_encoders.pkl
#   │   └── cutigo_label_encoders.pkl
#   └── data/
#       └── cutigo_master_places.csv
# ================================================================

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import random
import math
import joblib
import os
import warnings
import mysql.connector
import json as json_lib
warnings.filterwarnings("ignore")

app = Flask(__name__)

# ================================================================
# LOAD MODELS + DATA ON STARTUP
# ================================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "ml_model")
DATA_DIR    = os.path.join(BASE_DIR, "data")

print("[CutiGo] Loading ML models...")
rf_model    = joblib.load(os.path.join(MODEL_DIR, "cutigo_rf_model.pkl"))
feature_enc = joblib.load(os.path.join(MODEL_DIR, "cutigo_encoders.pkl"))
label_enc   = joblib.load(os.path.join(MODEL_DIR, "cutigo_label_encoders.pkl"))
le_cat      = label_enc["category"]
print("[CutiGo] ML models loaded OK")

print("[CutiGo] Loading places database...")
places = pd.read_csv(os.path.join(DATA_DIR, "cutigo_master_places.csv"))
if "place_name" in places.columns:
    places["display_name"] = places["place_name"]
elif "recommended_place" in places.columns:
    places["display_name"] = places["recommended_place"]
print(f"[CutiGo] {len(places):,} places loaded OK")

# ================================================================
# MYSQL CONNECTION (Railway)
# ----------------------------------------------------------------
# Reads connection details from environment variables - set these on
# the cutigo-api2 Railway service (Settings -> Variables), either by
# copy-pasting from the MySQL service's "Connect" tab, or using
# Railway's "Add Reference" option to auto-link the MySQL service:
#   MYSQLHOST, MYSQLPORT, MYSQLUSER, MYSQLPASSWORD, MYSQLDATABASE
# ================================================================

def get_db_connection():
    """Open a new MySQL connection using Railway's env vars."""
    return mysql.connector.connect(
        host=os.environ.get("MYSQLHOST"),
        port=int(os.environ.get("MYSQLPORT", 3306)),
        user=os.environ.get("MYSQLUSER"),
        password=os.environ.get("MYSQLPASSWORD"),
        database=os.environ.get("MYSQLDATABASE"),
    )


# ================================================================
# CONSTANTS
# ================================================================
INPUT_FEATURES = [
    "state", "budget_preference", "activity_interest",
    "trip_duration", "group_type",
    "transportation_preference", "accommodation_preference",
]

DURATION_TO_DAYS = {
    "Half Day": 0.5,
    "1 Day":    1,
    "2-3 Days": 3,
    "4-7 Days": 5,
    "1 Week+":  7,
}

TIME_SLOTS = {
    2: ["Morning (9:00 AM - 12:00 PM)", "Afternoon (2:00 PM - 5:00 PM)"],
    3: ["Morning (9:00 AM - 12:00 PM)", "Afternoon (2:00 PM - 5:00 PM)",
        "Evening (7:00 PM - 9:00 PM)"],
}

DAY_START_HOUR = 8.0   # 8:00 AM
DAY_END_HOUR = 22.0    # 10:00 PM
DAY_BUDGET_HOURS = DAY_END_HOUR - DAY_START_HOUR  # 14 hours/day to fill

DURATION_RANGE_BY_DESTINATION_TYPE = {
    "Beach & Island": (2.5, 4.0),
    "Highland & Nature": (2.0, 3.0),
    "Eco & Wildlife": (1.5, 2.5),
    "Heritage & Culture": (1.0, 1.5),
    "City & Urban": (1.0, 1.5),
    "Food & Culinary": (1.0, 1.5),
}
DEFAULT_DURATION_RANGE = (1.0, 1.5)

DURATION_RANGE_BY_CATEGORY = {
    "Shopping": (2.5, 3.0),
}

WATERPARK_ADVENTURE_DURATION_HOURS = 5.0

MAX_PLACES_PER_DAY = 6

MAX_SCAN_PER_ACTIVITY_PER_ROUND = 15

OPERATING_HOURS_BY_CATEGORY = {
    "Heritage & Museum":     (8.0, 18.0),
    "Religious & Cultural":  (8.0, 18.0),
    "Sightseeing & Tours":   (8.0, 21.0),
    "Shopping":              (8.0, 21.0),
    "Nature & Outdoors":     (8.0, 22.0),
    "Adventure & Sports":    (8.0, 22.0),
    "Entertainment":         (8.0, 22.0),
    "Food & Dining":         (8.0, 22.0),
}
DEFAULT_OPERATING_HOURS = (8.0, 20.0)


def get_operating_hours(category: str):
    return OPERATING_HOURS_BY_CATEGORY.get(category, DEFAULT_OPERATING_HOURS)


WATERPARK_ADVENTURE_KEYWORDS = [
    "water park", "waterpark", "adventure park", "theme park",
    "escape park", "skyway", "cable car", "luge",
]


def get_place_duration_hours(place_row) -> float:
    name = str(place_row.get("display_name", place_row.get("place_name", ""))).lower()
    if any(kw in name for kw in WATERPARK_ADVENTURE_KEYWORDS):
        return WATERPARK_ADVENTURE_DURATION_HOURS

    category = place_row.get("category", "")
    if category == "Adventure & Sports":
        return WATERPARK_ADVENTURE_DURATION_HOURS

    if category in DURATION_RANGE_BY_CATEGORY:
        low, high = DURATION_RANGE_BY_CATEGORY[category]
        return round(random.uniform(low, high), 1)

    dest_type = place_row.get("destination_type", "")
    low, high = DURATION_RANGE_BY_DESTINATION_TYPE.get(dest_type, DEFAULT_DURATION_RANGE)
    return round(random.uniform(low, high), 1)


def is_beach_or_island(place_row) -> bool:
    return place_row.get("destination_type", "") == "Beach & Island"


def format_time_range(start_hour: float, duration: float) -> str:
    end_hour = start_hour + duration

    def fmt(h):
        h = h % 24
        period = "AM" if h < 12 else "PM"
        display_h = int(h) if int(h) % 12 != 0 else 12
        display_h = display_h if display_h <= 12 else display_h - 12
        if display_h == 0:
            display_h = 12
        minutes = int(round((h - int(h)) * 60))
        return f"{display_h}:{minutes:02d} {period}"

    return f"{fmt(start_hour)} - {fmt(end_hour)}"


def sort_for_beach_preference(candidate_places: pd.DataFrame) -> pd.DataFrame:
    if candidate_places.empty:
        return candidate_places
    df = candidate_places.copy()
    df["_beach_pref"] = df.apply(is_beach_or_island, axis=1).astype(int)
    sort_columns = ["_fame_rank", "_beach_pref"]
    if "_iconic_rank" in df.columns:
        sort_columns.insert(0, "_iconic_rank")
    return df.sort_values(
        sort_columns,
        ascending=[False] * len(sort_columns),
        kind="stable",
    )

VALID_STATES = sorted(places["state"].unique().tolist())

STATE_TAGLINES = {
    "Johor":           {"title": "Discover Johor, Harimau Selatan",          "tagline": "Permata Selatan, warisan dan semangat juang"},
    "Kedah":           {"title": "Discover Kedah, Jelapang Padi",            "tagline": "Hamparan sawah padi dan keindahan Langkawi"},
    "Kelantan":        {"title": "Discover Kelantan, Serambi Mekah",         "tagline": "Tradisi, kraf tangan, dan budaya Pantai Timur"},
    "Kuala Lumpur":    {"title": "Discover Kuala Lumpur",                    "tagline": "Jantung bandar raya, gemerlap menara dan budaya"},
    "Melaka":          {"title": "Discover Melaka, Bandaraya Bersejarah",    "tagline": "The Historic State di tebing Selat Melaka"},
    "Negeri Sembilan": {"title": "Discover Negeri Sembilan, Negeri Beradat", "tagline": "Adat dan budaya yang masih dipelihara"},
    "Pahang":          {"title": "Discover Pahang, Negeri Tok Gajah",        "tagline": "Malaysia Truly Asia's Adventure Destination"},
    "Penang":          {"title": "Discover Penang, Pulau Mutiara",           "tagline": "Pearl of the Orient - seni, warisan, dan kuliner"},
    "Perak":           {"title": "Discover Perak, Darul Ridzuan",            "tagline": "Land of Grace - gua kapur dan bandar lama"},
    "Perlis":          {"title": "Discover Perlis, Indera Kayangan",         "tagline": "Negeri terkecil dengan keindahan luar bandar"},
    "Sabah":           {"title": "Discover Sabah, Negeri Di Bawah Bayu",     "tagline": "Land Below the Wind - gunung, hutan, dan laut"},
    "Sarawak":         {"title": "Discover Sarawak, Bumi Kenyalang",         "tagline": "Land of the Hornbills - gua purba dan budaya"},
    "Selangor":        {"title": "Discover Selangor",                       "tagline": "The Gateway to Malaysia - tema taman dan bandar"},
    "Terengganu":      {"title": "Discover Terengganu, Gerbang Pantai Timur", "tagline": "Negeri Warisan Pesisir Air yang jernih"},
}

DEFAULT_TAGLINE = {"title": "Your Malaysia Adventure", "tagline": "A trip built just for you"}

VALID_ACTIVITIES = [
    "Nature", "Sightseeing", "Culture", "Shopping",
    "Food", "Adventure", "Entertainment", "Relaxation"
]

ACTIVITY_PRIMARY_CATEGORIES = {
    "Nature":        "Nature & Outdoors",
    "Sightseeing":   "Sightseeing & Tours",
    "Culture":       "Heritage & Museum",
    "Shopping":      "Shopping",
    "Food":          "Food & Dining",
    "Adventure":     "Adventure & Sports",
    "Entertainment": "Entertainment",
}

ICONIC_PLACE_PRIORITY = {
    "the national museum of malaysia": 100,
    "central market": 100,
    "the exchange trx": 100,

    "petronas twin towers": 100,
    "batu caves": 95,
    "sipadan island": 93,
    "perhentian islands": 92,
    "redang island": 90,
    "taman negara": 88,
    "genting skyway": 85,
    "langkawi cable car": 84,
    "national mosque": 83,
    "george town street art": 82,
    "mabul island": 81,
    "penang hill heritage trail": 81,
    "crystal mosque": 80,
    "sarawak cultural village": 79,
    "poring hot springs": 78,
    "cherating beach": 78,
    "pangkor island": 77,
    "kapas island": 76,
    "pulau lang tengah": 73,
    "putra mosque": 72,
    "bohey dulang island": 71,
    "sky mirror kuala selangor": 70,
    "kilim geoforest park": 69,
    "melaka sultanate palace": 68,
    "portuguese settlement": 67,
    "sultan salahuddin abdul aziz mosque": 66,
    "endau rompin national park": 65,
    "maritime museum": 64,
    "labuan marine park": 63,
    "putrajaya lake": 62,
    "seri wawasan bridge": 60,
    "elephant sanctuary kuala gandah": 59,
    "perdana putra": 58,
    "jalan alor": 58,
    "puteri harbour": 57,
    "escape penang": 56,
    "chinatown kuala lumpur": 55,
    "zoo taiping": 54,
    "sekinchan paddy fields": 53,
    "taming sari tower": 52,
    "kuala selangor fireflies": 51,
    "mahsuri tomb": 50,
    "cruise tasik putrajaya": 50,
    "cape rachado lighthouse": 49,
    "pasar siti khadijah": 48,
    "putrajaya botanical garden": 47,
    "entopia butterfly farm": 47,
    "blue lagoon beach": 46,
    "tengku tengah zaharah mosque": 45,
    "frim": 45,
    "bukit merah laketown": 44,
    "muhammadi mosque": 44,
    "millennium monument": 43,
    "army museum port dickson": 43,
    "craft museum": 42,
    "austin heights water park": 42,
    "sungei lembing": 41,
    "anjung floria": 41,
    "terengganu drawbridge": 40,
    "perdana botanical gardens": 38,
    "wat photivihan": 37,
    "jeram toi waterfall": 36,
    "tasik melati": 35,
    "jelawang waterfall": 34,
    "bukit keteri": 33,
    "pulau papan": 33,
    "timah tasoh lake": 32,
    "pd ostrich farm": 31,
    "snake and reptile farm": 30,
    "alive 3d art gallery": 28,
    "bank kerapu": 27,
    "chimney museum": 26,
    "peace park": 25,
    "labuan war cemetery": 24,
    "surrender point": 23,
    "financial park": 20,
}

VALID_BUDGETS       = ["Budget", "Moderate", "Premium", "Luxury"]
VALID_DURATIONS     = ["Half Day", "1 Day", "2-3 Days", "4-7 Days", "1 Week+"]
VALID_GROUPS        = ["Solo", "Couple", "Family", "Group of Friends"]
VALID_TRANSPORTS    = ["Car/Self-Drive", "Tour Bus", "Public Transport",
                       "Taxi/Grab", "Flight", "Boat"]
VALID_ACCOMMODATIONS = [
    "Hostel", "Budget Hotel", "Homestay", "Mid-range Hotel",
    "Boutique Hotel", "Resort/Luxury Hotel", "Glamping/Camp"
]

# ================================================================
# HELPER FUNCTIONS
# ================================================================

def encode_input(user_prefs, activity_override=None):
    prefs = user_prefs.copy()
    if activity_override:
        prefs["activity_interest"] = activity_override
    encoded = []
    for col in INPUT_FEATURES:
        le  = feature_enc[col]
        val = str(prefs.get(col, ""))
        encoded.append(int(le.transform([val])[0]) if val in le.classes_ else 0)
    return np.array(encoded).reshape(1, -1)


def predict_category(user_prefs, activity):
    arr      = encode_input(user_prefs, activity_override=activity)
    idx      = rf_model.predict(arr)[0]
    proba    = rf_model.predict_proba(arr)[0]
    cat      = le_cat.inverse_transform([idx])[0]
    top3_idx = np.argsort(proba)[::-1][:3]
    top3     = [
        {"category": le_cat.classes_[i], "confidence": round(float(proba[i]) * 100, 1)}
        for i in top3_idx
    ]
    return cat, top3


def get_more_info_link(place_row, place_name, state):
    booking = str(place_row.get("booking_link", ""))
    if booking and booking not in ("nan", "", "None", "NaN"):
        if not booking.startswith("http"):
            booking = "https://" + booking
        return booking, "website"
    query = place_name.replace(" ", "+") + "+" + state.replace(" ", "+")
    return f"https://www.google.com/maps/search/{query}", "google_maps"


def _rank_by_fame_proxy(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["_iconic_rank"] = (
        df["display_name"].astype(str).str.strip().str.lower()
        .map(ICONIC_PLACE_PRIORITY).fillna(0).astype(int)
    )
    if "is_famous" in df.columns:
        df["_fame_rank"] = df["is_famous"].astype(int)
    else:
        df["_fame_rank"] = (df["source"] == "Google Places").astype(int)
    return df.sort_values(
        ["_iconic_rank", "_fame_rank"],
        ascending=[False, False],
        kind="stable",
    )


def query_places_for_activity(user_state, predicted_cat, top3_cats,
                               n_needed, exclude_names=None):
    if exclude_names is None:
        exclude_names = set()

    pool_target = min(max(n_needed * 3, 12), 60)

    all_cands = pd.DataFrame()

    p1 = places[
        (places["state"] == user_state) &
        (places["category"] == predicted_cat) &
        (~places["display_name"].isin(exclude_names))
    ].copy()
    p1["match_quality"] = "exact"
    all_cands = pd.concat([all_cands, p1])

    if len(all_cands) < pool_target:
        for item in top3_cats[1:]:
            cat = item["category"]
            p2  = places[
                (places["state"] == user_state) &
                (places["category"] == cat) &
                (~places["display_name"].isin(
                    set(all_cands["display_name"]) | exclude_names
                ))
            ].copy()
            p2["match_quality"] = "related"
            all_cands = pd.concat([all_cands, p2])
            if len(all_cands) >= pool_target:
                break

    if len(all_cands) == 0:
        return pd.DataFrame()

    all_cands = all_cands[all_cands["state"] == user_state].copy()
    if all_cands.empty:
        return pd.DataFrame()

    all_cands = all_cands.drop_duplicates(subset=["display_name"])
    all_cands = _rank_by_fame_proxy(all_cands)
    all_cands = sort_for_beach_preference(all_cands)

    return all_cands.head(pool_target).reset_index(drop=True)


def split_days(total_days, activities):
    n         = len(activities)
    base      = total_days // n
    remainder = total_days % n
    return {act: base + (1 if i < remainder else 0)
            for i, act in enumerate(activities)}


def allocate_places_by_duration(candidates: pd.DataFrame, day_budget_hours: float = DAY_BUDGET_HOURS,
                                 start_hour: float = DAY_START_HOUR, places_used_so_far: int = 0):
    chosen = []
    used_indices = []
    current_hour = start_hour
    remaining_budget = max(0.0, day_budget_hours - (start_hour - DAY_START_HOUR))
    places_used_total = places_used_so_far

    for idx, row in candidates.iterrows():
        if places_used_total >= MAX_PLACES_PER_DAY:
            break
        if row.get("category") == "Accommodation":
            continue

        duration = get_place_duration_hours(row)
        category = row.get("category", "")
        open_hour, close_hour = get_operating_hours(category)

        effective_start = max(current_hour, open_hour)
        finish_hour = effective_start + duration

        if effective_start >= close_hour or finish_hour > close_hour:
            continue
        if duration > remaining_budget:
            continue

        chosen.append((row, effective_start, duration))
        used_indices.append(idx)
        current_hour = finish_hour
        remaining_budget -= duration
        places_used_total += 1
        if remaining_budget <= 0.5:
            break

    remaining_candidates = candidates.drop(index=used_indices)
    return chosen, remaining_candidates, current_hour, remaining_budget, places_used_total


def pick_filler_stop(user_state: str, current_hour: float, remaining_budget: float, exclude_names: set):
    if remaining_budget < 1.0:
        return None, current_hour

    food_candidates = places[
        (places["state"] == user_state) &
        (places["category"] == "Food & Dining")
    ].copy()
    if food_candidates.empty:
        return None, current_hour

    open_hour, close_hour = get_operating_hours("Food & Dining")
    if current_hour >= close_hour:
        return None, current_hour

    food_candidates = food_candidates[~food_candidates["display_name"].isin(exclude_names)]
    if food_candidates.empty:
        return None, current_hour

    food_candidates = _rank_by_fame_proxy(food_candidates)
    chosen_row = food_candidates.iloc[0]
    duration = min(1.5, remaining_budget, close_hour - current_hour)
    if duration < 0.5:
        return None, current_hour

    return (chosen_row, current_hour, duration), current_hour + duration


def pick_accommodation_checkin(user_prefs: dict, user_state: str, exclude_names: set):
    wants_accommodation_highlight = (
        user_prefs.get("activity_interest") == "Relaxation" or
        user_prefs.get("accommodation_preference") in ("Resort/Luxury Hotel", "Boutique Hotel")
    )
    if not wants_accommodation_highlight:
        return None

    candidates = places[
        (places["state"] == user_state) &
        (places["category"] == "Accommodation") &
        (~places["display_name"].isin(exclude_names))
    ].copy()
    if candidates.empty:
        return None

    candidates = _rank_by_fame_proxy(candidates)
    return candidates.iloc[0]


def validate_request(data):
    required = [
        "state", "budget_preference", "trip_duration",
        "group_type", "transportation_preference",
        "accommodation_preference", "activity_interest"
    ]
    for field in required:
        if field not in data:
            return False, f"Missing required field: '{field}'"

    if data["state"] not in VALID_STATES:
        return False, f"Invalid state. Choose from: {VALID_STATES}"

    if data["budget_preference"] not in VALID_BUDGETS:
        return False, f"Invalid budget. Choose from: {VALID_BUDGETS}"

    if data["trip_duration"] not in VALID_DURATIONS:
        return False, f"Invalid duration. Choose from: {VALID_DURATIONS}"

    if data["group_type"] not in VALID_GROUPS:
        return False, f"Invalid group_type. Choose from: {VALID_GROUPS}"

    if data["transportation_preference"] not in VALID_TRANSPORTS:
        return False, f"Invalid transportation. Choose from: {VALID_TRANSPORTS}"

    if data["accommodation_preference"] not in VALID_ACCOMMODATIONS:
        return False, f"Invalid accommodation. Choose from: {VALID_ACCOMMODATIONS}"

    activities = data["activity_interest"]
    if isinstance(activities, str):
        activities = [a.strip() for a in activities.split(",")]
    for act in activities:
        if act not in VALID_ACTIVITIES:
            return False, f"Invalid activity: '{act}'. Choose from: {VALID_ACTIVITIES}"

    return True, ""


# ================================================================
# API ENDPOINTS
# ================================================================

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status":  "ok",
        "app":     "CutiGo Trip Matching API",
        "version": "1.0",
        "places":  len(places),
        "states":  len(VALID_STATES),
    })


@app.route("/states", methods=["GET"])
def get_states():
    return jsonify({
        "status": "ok",
        "states": VALID_STATES
    })


@app.route("/activities", methods=["GET"])
def get_activities():
    return jsonify({
        "status":     "ok",
        "activities": VALID_ACTIVITIES
    })


@app.route("/options", methods=["GET"])
def get_options():
    return jsonify({
        "status": "ok",
        "options": {
            "states":           VALID_STATES,
            "activities":       VALID_ACTIVITIES,
            "budgets":          VALID_BUDGETS,
            "durations":        VALID_DURATIONS,
            "groups":           VALID_GROUPS,
            "transports":       VALID_TRANSPORTS,
            "accommodations":   VALID_ACCOMMODATIONS,
        }
    })


# ================================================================
# MYSQL ENDPOINTS - Places search + Saved Trips CRUD
# ================================================================

@app.route("/places", methods=["GET"])
def search_places():
    """
    Search/filter places stored in MySQL (separate from the CSV used
    by /recommend - this is for the Android "search place to add/edit
    a trip slot" feature).

    Query params:
      name   (optional) - partial match on place_name, e.g. ?name=pantai
      state  (optional) - exact match on state, e.g. ?state=Sabah
      limit  (optional) - max rows to return, default 30, max 100
    """
    name = request.args.get("name", "").strip()
    state = request.args.get("state", "").strip()
    limit = min(int(request.args.get("limit", 30)), 100)

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    query = "SELECT * FROM places WHERE 1=1"
    params = []

    if name:
        query += " AND place_name LIKE %s"
        params.append(f"%{name}%")

    if state:
        query += " AND state = %s"
        params.append(state)

    query += " ORDER BY is_famous DESC, rating_imputed DESC LIMIT %s"
    params.append(limit)

    cursor.execute(query, params)
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    return jsonify({
        "status": "ok",
        "count": len(results),
        "places": results,
    })


@app.route("/famous-places", methods=["GET"])
def get_famous_places():
    """Top N is_famous places (highest rated first), for the homepage carousel."""
    limit = min(int(request.args.get("limit", 15)), 50)

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT * FROM places WHERE is_famous = 1 "
        "ORDER BY rating_imputed DESC LIMIT %s",
        (limit,)
    )
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    return jsonify({"status": "ok", "count": len(results), "places": results})


@app.route("/trips", methods=["POST"])
def save_trip():
    data = request.get_json()

    if not data or "user_id" not in data:
        return jsonify({"status": "error", "message": "user_id is required"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO saved_trips
            (user_id, state, title, tagline, budget_preference, trip_duration,
             actual_days, group_type, transportation_preference,
             accommodation_preference, activity_interest, itinerary_json)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            data.get("user_id"),
            data.get("state"),
            data.get("title"),
            data.get("tagline"),
            data.get("budget_preference"),
            data.get("trip_duration"),
            data.get("actual_days"),
            data.get("group_type"),
            data.get("transportation_preference"),
            data.get("accommodation_preference"),
            # activity_interest can arrive as a list ["Nature","Food"] or
            # a pre-joined string "Nature, Food" - normalise to string.
            (", ".join(data["activity_interest"])
             if isinstance(data.get("activity_interest"), list)
             else str(data.get("activity_interest", ""))),
            json_lib.dumps(data.get("itinerary", [])),
        )
    )
    conn.commit()
    new_trip_id = cursor.lastrowid
    cursor.close()
    conn.close()

    return jsonify({"status": "ok", "trip_id": new_trip_id})


@app.route("/trips", methods=["GET"])
def get_trips():
    user_id = request.args.get("user_id", "").strip()

    if not user_id:
        return jsonify({"status": "error", "message": "user_id is required"}), 400

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT * FROM saved_trips WHERE user_id = %s ORDER BY created_at DESC",
        (user_id,)
    )
    trips = cursor.fetchall()
    cursor.close()
    conn.close()

    for trip in trips:
        if trip.get("itinerary_json"):
            trip["itinerary"] = json_lib.loads(trip["itinerary_json"])
            del trip["itinerary_json"]
        if trip.get("created_at"):
            trip["created_at"] = str(trip["created_at"])

    return jsonify({"status": "ok", "count": len(trips), "trips": trips})


@app.route("/trips/<int:trip_id>", methods=["PUT"])
def update_trip(trip_id):
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "No data provided"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    allowed_fields = {
        "state": "state", "title": "title", "tagline": "tagline",
        "budget_preference": "budget_preference", "trip_duration": "trip_duration",
        "actual_days": "actual_days", "group_type": "group_type",
        "transportation_preference": "transportation_preference",
        "accommodation_preference": "accommodation_preference",
    }
    set_parts = []
    params = []
    for key, column in allowed_fields.items():
        if key in data:
            set_parts.append(f"{column} = %s")
            params.append(data[key])

    if "activity_interest" in data:
        set_parts.append("activity_interest = %s")
        params.append(", ".join(data["activity_interest"]))

    if "itinerary" in data:
        set_parts.append("itinerary_json = %s")
        params.append(json_lib.dumps(data["itinerary"]))

    if not set_parts:
        cursor.close()
        conn.close()
        return jsonify({"status": "error", "message": "No valid fields to update"}), 400

    params.append(trip_id)
    query = f"UPDATE saved_trips SET {', '.join(set_parts)} WHERE trip_id = %s"
    cursor.execute(query, params)
    conn.commit()
    affected = cursor.rowcount
    cursor.close()
    conn.close()

    if affected == 0:
        return jsonify({"status": "error", "message": "Trip not found"}), 404

    return jsonify({"status": "ok", "trip_id": trip_id, "updated": True})


@app.route("/trips/<int:trip_id>", methods=["DELETE"])
def delete_trip(trip_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM saved_trips WHERE trip_id = %s", (trip_id,))
    conn.commit()
    affected = cursor.rowcount
    cursor.close()
    conn.close()

    if affected == 0:
        return jsonify({"status": "error", "message": "Trip not found"}), 404

    return jsonify({"status": "ok", "trip_id": trip_id, "deleted": True})


# ================================================================
# MAIN RECOMMEND ENDPOINT (unchanged logic from the previous version)
# ================================================================

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "No JSON body received"}), 400

    is_valid, err = validate_request(data)
    if not is_valid:
        return jsonify({"status": "error", "message": err}), 400

    raw_activities = data["activity_interest"]
    if isinstance(raw_activities, str):
        activities = [a.strip() for a in raw_activities.split(",") if a.strip()]
    else:
        activities = [a.strip() for a in raw_activities if a.strip()]
    activities = [a for a in activities if a in VALID_ACTIVITIES]
    if not activities:
        activities = ["Nature"]

    user_prefs = {
        "state":                     data["state"],
        "budget_preference":         data["budget_preference"],
        "activity_interest":         activities[0],
        "trip_duration":             data["trip_duration"],
        "group_type":                data["group_type"],
        "transportation_preference": data["transportation_preference"],
        "accommodation_preference":  data["accommodation_preference"],
    }

    duration   = data["trip_duration"]
    actual_days = data.get("actual_days")

    is_half = (duration == "Half Day")

    if is_half:
        total_int = 1
    elif actual_days is not None:
        try:
            total_int = max(1, int(actual_days))
        except (TypeError, ValueError):
            total_int = int(DURATION_TO_DAYS.get(duration, 1))
    else:
        total_days = DURATION_TO_DAYS.get(duration, 1)
        total_int = 1 if total_days == 0.5 else int(total_days)

    allocation = {}
    day_activity_map = {day: activities for day in range(1, total_int + 1)}

    act_preds = {}
    ai_predictions = []
    for act in activities:
        cat, top3 = predict_category(user_prefs, act)
        act_preds[act] = {"category": cat, "top3": top3}
        ai_predictions.append({
            "activity":           act,
            "predicted_category": cat,
            "confidence":         top3[0]["confidence"],
            "top3":               top3,
        })

    used       = set()
    act_pools  = {}
    for act in activities:
        cat   = act_preds[act]["category"]
        top3  = act_preds[act]["top3"]
        n_need = max(total_int * MAX_PLACES_PER_DAY * 2, 20)
        query_cat = ACTIVITY_PRIMARY_CATEGORIES.get(act, cat)
        query_top3 = [{"category": query_cat, "confidence": 100.0}]
        query_top3.extend(
            item for item in top3
            if item["category"] != query_cat
        )

        matched = query_places_for_activity(
            user_state    = data["state"],
            predicted_cat = query_cat,
            top3_cats     = query_top3,
            n_needed      = n_need,
            exclude_names = used,
        )
        act_pools[act] = matched

    itinerary  = []
    all_places = []

    for day_num in sorted(day_activity_map.keys()):
        day_acts  = day_activity_map[day_num]
        day_label = "HALF DAY" if is_half else f"DAY {day_num}"

        day_places_list = []
        day_used_names = set()

        day_current_hour = DAY_START_HOUR
        day_remaining_budget = 7.0 if is_half else DAY_BUDGET_HOURS
        day_places_used = 0

        act_pointers = {act: 0 for act in day_acts}
        act_rows = {act: act_pools[act].reset_index(drop=True) for act in day_acts}
        act_deferred_rows = {act: [] for act in day_acts}

        days_remaining = total_int - day_num + 1
        remaining_unique_names = set()
        for act in day_acts:
            for _, candidate in act_rows[act].iterrows():
                candidate_name = str(candidate.get("display_name", ""))
                candidate_state = str(candidate.get("state", "")).strip()
                if (
                    candidate_name and
                    candidate_name not in used and
                    candidate_state == data["state"] and
                    candidate.get("category") != "Accommodation"
                ):
                    remaining_unique_names.add(candidate_name)

        available_unique_count = len(remaining_unique_names)
        if available_unique_count:
            day_place_limit = min(
                MAX_PLACES_PER_DAY,
                max(1, math.ceil(available_unique_count / days_remaining)),
            )
        else:
            day_place_limit = 0

        max_rounds = MAX_PLACES_PER_DAY + 2
        for _round in range(max_rounds):

            if day_places_used >= day_place_limit or day_remaining_budget <= 0.5:
                break

            made_progress_this_round = False

            for act in day_acts:

                if day_places_used >= day_place_limit or day_remaining_budget <= 0.5:
                    break

                rows = act_rows[act]
                ptr = act_pointers[act]

                placed = False
                scanned_this_round = 0
                while (
                    ptr < len(rows) and
                    scanned_this_round < MAX_SCAN_PER_ACTIVITY_PER_ROUND
                ):

                    candidate_row = rows.iloc[ptr]
                    ptr += 1
                    scanned_this_round += 1

                    if candidate_row.get("category") == "Accommodation":
                        continue

                    name = str(candidate_row.get("display_name", ""))
                    if name in used or name in day_used_names:
                        continue

                    candidate_state = str(candidate_row.get("state", "")).strip()
                    if candidate_state != data["state"]:
                        continue

                    duration_h = get_place_duration_hours(candidate_row)
                    category = candidate_row.get("category", "")
                    open_hour, close_hour = get_operating_hours(category)

                    effective_start = max(day_current_hour, open_hour)
                    finish_hour = effective_start + duration_h

                    if effective_start >= close_hour or finish_hour > close_hour:
                        act_deferred_rows[act].append(candidate_row)
                        continue
                    if duration_h > day_remaining_budget:
                        act_deferred_rows[act].append(candidate_row)
                        continue

                    state = candidate_state
                    link, link_type = get_more_info_link(candidate_row, name, state)
                    place_obj = {
                        "time_slot":     format_time_range(effective_start, duration_h),
                        "place_name":    name,
                        "category":      str(candidate_row.get("category", "")),
                        "duration_hours": duration_h,
                        "rating":        round(float(candidate_row.get("rating_imputed", 0)), 1),
                        "more_info_url": link,
                        "link_type":     link_type,
                        "match_quality": str(candidate_row.get("match_quality", "")),
                        "state":         state,
                    }
                    day_places_list.append(place_obj)
                    all_places.append(place_obj)
                    day_used_names.add(name)
                    used.add(name)

                    day_current_hour = finish_hour
                    day_remaining_budget -= duration_h
                    day_places_used += 1

                    placed = True
                    made_progress_this_round = True
                    break

                act_pointers[act] = ptr

            if not made_progress_this_round:
                break

        for act in day_acts:
            used_ptr = act_pointers[act]
            deferred_df = pd.DataFrame(act_deferred_rows[act])
            untried_df = act_rows[act].iloc[used_ptr:]
            carried = pd.concat([deferred_df, untried_df], ignore_index=True)
            if not carried.empty:
                carried = carried.drop_duplicates(
                    subset=["display_name"], keep="first"
                )
            act_pools[act] = carried.reset_index(drop=True)

        filler, _ = pick_filler_stop(
            user_state=data["state"],
            current_hour=day_current_hour,
            remaining_budget=day_remaining_budget,
            exclude_names=used | day_used_names,
        )
        if filler is not None:
            f_row, f_start, f_duration = filler
            f_name = str(f_row.get("display_name", ""))
            f_state = str(f_row.get("state", data["state"]))
            f_link, f_link_type = get_more_info_link(f_row, f_name, f_state)
            filler_obj = {
                "time_slot":     format_time_range(f_start, f_duration),
                "place_name":    f_name,
                "category":      "Food & Dining",
                "duration_hours": f_duration,
                "rating":        round(float(f_row.get("rating_imputed", 0)), 1),
                "more_info_url": f_link,
                "link_type":     f_link_type,
                "match_quality": "filler",
                "state":         f_state,
            }
            day_places_list.append(filler_obj)
            all_places.append(filler_obj)
            day_used_names.add(f_name)
            used.add(f_name)

        if day_num == 1:
            primary_act = day_acts[0]
            checkin_row = pick_accommodation_checkin(
                user_prefs={**user_prefs, "activity_interest": primary_act},
                user_state=data["state"],
                exclude_names=used | day_used_names,
            )
            if checkin_row is not None:
                name = str(checkin_row.get("display_name", ""))
                state = str(checkin_row.get("state", data["state"]))
                link, link_type = get_more_info_link(checkin_row, name, state)
                checkin_obj = {
                    "time_slot":     "Check-in (6:00 PM onwards)",
                    "place_name":    name,
                    "category":      "Accommodation",
                    "duration_hours": None,
                    "rating":        round(float(checkin_row.get("rating_imputed", 0)), 1),
                    "more_info_url": link,
                    "link_type":     link_type,
                    "match_quality": "accommodation",
                    "state":         state,
                }
                day_places_list.append(checkin_obj)
                all_places.append(checkin_obj)
                used.add(name)

        if len(day_acts) == 1:
            day_activity_label = day_acts[0]
            day_predicted_category = act_preds[day_acts[0]]["category"]
        else:
            day_activity_label = " & ".join(day_acts)
            day_predicted_category = " & ".join(
                sorted(set(act_preds[a]["category"] for a in day_acts))
            )

        itinerary.append({
            "day":                day_num,
            "day_label":          day_label,
            "activity":           day_activity_label,
            "predicted_category": day_predicted_category,
            "places":             day_places_list,
        })

    avg_rating = round(float(np.mean([p["rating"] for p in all_places])), 2) if all_places else 0.0
    day_alloc  = [
        {"activity": act, "days": n, "category": act_preds[act]["category"]}
        for act, n in allocation.items()
    ] if allocation else []

    state_tagline = STATE_TAGLINES.get(data["state"], DEFAULT_TAGLINE)

    trip_summary = {
        "state":          data["state"],
        "title":          state_tagline["title"],
        "tagline":        state_tagline["tagline"],
        "duration":       duration,
        "total_days":     total_int,
        "activities":     activities,
        "day_allocation": day_alloc,
        "total_places":   len(all_places),
        "avg_rating":     avg_rating,
        "group":          data["group_type"],
        "budget":         data["budget_preference"],
        "transport":      data["transportation_preference"],
        "accommodation":  data["accommodation_preference"],
    }

    return jsonify({
        "status":           "ok",
        "user_preferences": data,
        "ai_predictions":   ai_predictions,
        "trip_summary":     trip_summary,
        "itinerary":        itinerary,
    })


# ================================================================
# RUN
# ================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
