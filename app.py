# ================================================================
# CutiGo: Flask API — app.py
# Deploy to Render (render.com)
#
# Endpoints:
#   GET  /              → health check
#   POST /recommend     → main itinerary endpoint
#   GET  /states        → list all available states
#   GET  /activities    → list all activity interests
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
import joblib
import os
import warnings
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
print("[CutiGo] ML models loaded ✅")

print("[CutiGo] Loading places database...")
places = pd.read_csv(os.path.join(DATA_DIR, "cutigo_master_places.csv"))
if "place_name" in places.columns:
    places["display_name"] = places["place_name"]
elif "recommended_place" in places.columns:
    places["display_name"] = places["recommended_place"]
print(f"[CutiGo] {len(places):,} places loaded ✅")

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
    2: ["Morning (9:00 AM – 12:00 PM)", "Afternoon (2:00 PM – 5:00 PM)"],
    3: ["Morning (9:00 AM – 12:00 PM)", "Afternoon (2:00 PM – 5:00 PM)",
        "Evening (7:00 PM – 9:00 PM)"],
}

# ================================================================
# SMART DURATION ALLOCATION
# ----------------------------------------------------------------
# Instead of forcing every place into an equal-size slot, each place
# gets a duration (in hours) based on its destination_type/category -
# places where people typically linger (beaches, islands, rivers,
# waterparks/adventure) get more time than quick-visit spots
# (heritage sites, city landmarks, food stops).
# ================================================================

DAY_START_HOUR = 8.0   # 8:00 AM
DAY_END_HOUR = 22.0    # 10:00 PM
DAY_BUDGET_HOURS = DAY_END_HOUR - DAY_START_HOUR  # 14 hours/day to fill

# Duration in hours by destination_type - the main driver of "how long
# people linger" (beach/island/nature spots get more time than a quick
# heritage/city stop).
DURATION_BY_DESTINATION_TYPE = {
    "Beach & Island": 3.0,       # pantai/island - lepak lama
    "Highland & Nature": 2.5,    # includes rivers/waterfalls - lepak lama
    "Eco & Wildlife": 2.0,
    "Adventure & Sports": 4.0,   # half-day style (waterpark, adventure parks)
    "Heritage & Culture": 1.5,
    "City & Urban": 1.5,
    "Food & Culinary": 1.5,
}
DEFAULT_DURATION_HOURS = 1.5

# Maximum number of "main activity" places per day - once this is hit,
# any remaining time in the day budget is used for a food/relax stop
# instead of cramming in more sightseeing (e.g. 9 museums in one day
# isn't realistic, even if each one is individually short).
MAX_PLACES_PER_DAY = 6

# Operating hours by category - heritage sites/museums/religious sites
# typically close in the late afternoon/early evening, while
# entertainment, beaches, and food venues commonly stay open into the
# evening. A place's start time is clamped so it can't begin after its
# category's closing hour, and the visit is skipped for that slot if
# it wouldn't finish before closing.
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

# Some places are tagged Entertainment/Sightseeing but ARE waterparks/
# adventure parks by name - these should also get the long "half-day"
# duration even though their destination_type/category doesn't say
# "Adventure & Sports".
WATERPARK_ADVENTURE_KEYWORDS = [
    "water park", "waterpark", "adventure park", "theme park",
    "escape park", "skyway", "cable car", "luge",
]


def get_place_duration_hours(place_row) -> float:
    """How many hours a typical visitor spends at this place."""
    name = str(place_row.get("display_name", place_row.get("place_name", ""))).lower()
    if any(kw in name for kw in WATERPARK_ADVENTURE_KEYWORDS):
        return 4.0
    dest_type = place_row.get("destination_type", "")
    return DURATION_BY_DESTINATION_TYPE.get(dest_type, DEFAULT_DURATION_HOURS)


def is_beach_or_island(place_row) -> bool:
    return place_row.get("destination_type", "") == "Beach & Island"


def format_time_range(start_hour: float, duration: float) -> str:
    """Format a start hour + duration into a 'H:MM AM/PM – H:MM AM/PM' string."""
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

    return f"{fmt(start_hour)} – {fmt(end_hour)}"


def sort_for_beach_preference(candidate_places: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder candidates so Beach & Island places are pulled toward the
    front of the queue (so they tend to land in the morning/sunrise
    slot first) - WITHOUT removing or hard-forcing anything. If a day
    is already full, a beach place simply lands wherever there's room;
    this is a soft preference, not a strict scheduling constraint.
    """
    if candidate_places.empty:
        return candidate_places
    df = candidate_places.copy()
    df["_beach_pref"] = df.apply(is_beach_or_island, axis=1).astype(int)
    # keep existing fame-based order as the primary sort, beach
    # preference as a secondary nudge within the same fame tier
    return df.sort_values(["_fame_rank", "_beach_pref"], ascending=[False, False], kind="stable")

VALID_STATES = sorted(places["state"].unique().tolist())

VALID_ACTIVITIES = [
    "Nature", "Sightseeing", "Culture", "Shopping",
    "Food", "Adventure", "Entertainment", "Relaxation"
]

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
    """Encode user preferences dict into numpy array."""
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
    """Run RF model for one activity, return top-3 category predictions."""
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
    """Return best available info link for a place."""
    booking = str(place_row.get("booking_link", ""))
    if booking and booking not in ("nan", "", "None", "NaN"):
        if not booking.startswith("http"):
            booking = "https://" + booking
        return booking, "website"
    # Fallback: Google Maps search
    query = place_name.replace(" ", "+") + "+" + state.replace(" ", "+")
    return f"https://www.google.com/maps/search/{query}", "google_maps"


def _rank_by_fame_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Order candidate places by a 'famous' proxy instead of rating
    (ratings in this dataset are inconsistent/unreliable). A place is
    treated as 'famous' if EITHER:
      - it matched a curated anchor list of well-known Malaysia
        landmarks (Petronas Twin Towers, Batu Caves, Cameron
        Highlands, etc. - see is_famous column, built from
        travel-authority sources), OR
      - it has source == 'Google Places' (implies a verified
        business presence, more reliable than OSM's crowdsourced
        tags)
    Famous places are shown first; within the same fame tier,
    original row order is kept rather than re-sorting by rating.
    """
    if df.empty:
        return df
    df = df.copy()
    if "is_famous" in df.columns:
        df["_fame_rank"] = df["is_famous"].astype(int)
    else:
        # fallback if is_famous column isn't present in the CSV yet
        df["_fame_rank"] = (df["source"] == "Google Places").astype(int)
    return df.sort_values("_fame_rank", ascending=False, kind="stable")


def query_places_for_activity(user_state, predicted_cat, top3_cats,
                               n_needed, exclude_names=None):
    """
    Query places DB with fallback strategy:
    1. Same state + predicted category
    2. Same state + related categories
    3. Nationwide + predicted category

    Within each tier, places are ranked by is_famous (Google Places /
    anchor-list landmarks preferred) as a 'famous/established' proxy -
    NOT by rating, since ratings in this dataset are inconsistent.
    Beach & Island places get a soft secondary preference within the
    same fame tier (see sort_for_beach_preference), since the smart
    itinerary builder tries to place them in the morning/evening slot.

    n_needed is used as a guide for how big a candidate pool to fetch,
    NOT a hard cap on the final count - the smart time-budget allocator
    (allocate_places_by_duration) decides how many of these candidates
    actually fit into a day, since each place can now take a different
    amount of time.
    """
    if exclude_names is None:
        exclude_names = set()

    # Fetch a larger pool than n_needed since each place may only take
    # 1.5-4 hours - we want enough options for the day-filling step to
    # choose from, especially once long-duration places (beach/island/
    # adventure) start eating into the day's time budget.
    pool_target = max(n_needed * 3, 12)

    all_cands = pd.DataFrame()

    # Priority 1: exact state + predicted category
    p1 = places[
        (places["state"] == user_state) &
        (places["category"] == predicted_cat) &
        (~places["display_name"].isin(exclude_names))
    ].copy()
    p1["match_quality"] = "exact"
    all_cands = pd.concat([all_cands, p1])

    # Priority 2: same state + related categories
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

    # Priority 3: nationwide fallback
    if len(all_cands) < pool_target:
        p3 = places[
            (places["category"] == predicted_cat) &
            (~places["display_name"].isin(
                set(all_cands["display_name"]) | exclude_names
            ))
        ].copy()
        p3["match_quality"] = "nationwide"
        all_cands = pd.concat([all_cands, p3])

    if len(all_cands) == 0:
        return pd.DataFrame()

    all_cands = all_cands.drop_duplicates(subset=["display_name"])
    all_cands = _rank_by_fame_proxy(all_cands)
    all_cands = sort_for_beach_preference(all_cands)

    return all_cands.head(pool_target).reset_index(drop=True)


def split_days(total_days, activities):
    """Split days equally across activities."""
    n         = len(activities)
    base      = total_days // n
    remainder = total_days % n
    return {act: base + (1 if i < remainder else 0)
            for i, act in enumerate(activities)}


def allocate_places_by_duration(candidates: pd.DataFrame, day_budget_hours: float = DAY_BUDGET_HOURS):
    """
    Fill one day's time budget by walking down the (already fame +
    beach-preference sorted) candidate list, taking each place's own
    estimated duration (get_place_duration_hours) until either the
    day's hours run out or MAX_PLACES_PER_DAY main-activity places
    have been scheduled (whichever comes first) - cramming in 8-9
    quick museum visits in one day isn't realistic even if each is
    individually short.

    Each place is also checked against its category's operating hours
    (get_operating_hours): if starting it now would mean visiting
    after closing time, or finishing after closing time, it's skipped
    for this day (it stays in the candidate pool for a future day).

    Accommodation-category places are NOT consumed from the day budget
    here - they're handled separately as a "check-in" entry appended
    at the end of the day (see pick_accommodation_checkin).

    Returns (chosen_rows: list of (row, start_hour, duration) tuples,
    remaining_candidates: DataFrame of whatever wasn't used, so the
    next day/activity can continue from where this one left off).
    """
    chosen = []
    used_indices = []
    current_hour = DAY_START_HOUR
    remaining_budget = day_budget_hours

    for idx, row in candidates.iterrows():
        if len(chosen) >= MAX_PLACES_PER_DAY:
            break
        if row.get("category") == "Accommodation":
            continue  # handled separately as a check-in entry

        duration = get_place_duration_hours(row)
        category = row.get("category", "")
        open_hour, close_hour = get_operating_hours(category)

        # Can't start before opening (shouldn't happen since the day
        # itself starts at DAY_START_HOUR, but kept for clarity/safety)
        effective_start = max(current_hour, open_hour)
        finish_hour = effective_start + duration

        if effective_start >= close_hour or finish_hour > close_hour:
            continue  # would still be open past closing - skip for today
        if duration > remaining_budget:
            continue  # doesn't fit in what's left of the day budget

        chosen.append((row, effective_start, duration))
        used_indices.append(idx)
        current_hour = finish_hour
        remaining_budget -= duration
        if remaining_budget <= 0.5:
            break

    remaining_candidates = candidates.drop(index=used_indices)
    return chosen, remaining_candidates, current_hour, remaining_budget


def pick_filler_stop(user_state: str, current_hour: float, remaining_budget: float, exclude_names: set):
    """
    If the day still has reasonable time left after hitting
    MAX_PLACES_PER_DAY (or running out of suitable candidates) but
    not enough to justify squeezing in another sightseeing stop, fill
    it with a Food & Dining place instead - a relaxed meal/coffee
    stop is a more realistic way to spend that time than cramming in
    one more attraction.
    """
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
    """
    If the user's preferences suggest they care about where they stay
    (Relaxation interest, or a Resort/Boutique-tier accommodation
    preference), pick one Accommodation-category place in their state
    to show as a "Check-in" entry at the end of the day. Returns None
    if accommodation isn't relevant for this user, or no match exists.
    """
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
    """Validate incoming request fields. Returns (is_valid, error_message)."""
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

    # activity_interest can be string or list
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
    """Health check endpoint."""
    return jsonify({
        "status":  "ok",
        "app":     "CutiGo Trip Matching API",
        "version": "1.0",
        "places":  len(places),
        "states":  len(VALID_STATES),
    })


@app.route("/states", methods=["GET"])
def get_states():
    """Return all available states."""
    return jsonify({
        "status": "ok",
        "states": VALID_STATES
    })


@app.route("/activities", methods=["GET"])
def get_activities():
    """Return all available activity interests."""
    return jsonify({
        "status":     "ok",
        "activities": VALID_ACTIVITIES
    })


@app.route("/options", methods=["GET"])
def get_options():
    """Return all valid options for every preference field."""
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


@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Main recommendation endpoint.
    """
    # ── Parse request ─────────────────────────────────────────────
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "No JSON body received"}), 400

    # ── Validate ──────────────────────────────────────────────────
    is_valid, err = validate_request(data)
    if not is_valid:
        return jsonify({"status": "error", "message": err}), 400

    # ── Normalise activities ──────────────────────────────────────
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
        "activity_interest":         activities[0],  # for encoding
        "trip_duration":             data["trip_duration"],
        "group_type":                data["group_type"],
        "transportation_preference": data["transportation_preference"],
        "accommodation_preference":  data["accommodation_preference"],
    }

    # ── Days calculation ──────────────────────────────────────────
    duration   = data["trip_duration"]
    total_days = DURATION_TO_DAYS.get(duration, 1)
    is_half    = (total_days == 0.5)
    total_int  = 1 if is_half else int(total_days)

    # ── Day → activity mapping ────────────────────────────────────
    allocation = {}
    if is_half or total_int == 1:
        day_activity_map = {1: activities}
    else:
        allocation = split_days(total_int, activities)
        day_activity_map = {}
        day_num = 1
        for act, n_days in allocation.items():
            for _ in range(n_days):
                day_activity_map[day_num] = [act]
                day_num += 1

    # ── ML Predictions ────────────────────────────────────────────
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

    # ── Query places (fetch a generous pool per activity) ─────────
    used       = set()
    act_pools  = {}
    for act in activities:
        cat   = act_preds[act]["category"]
        top3  = act_preds[act]["top3"]
        days_for_act = 1 if (is_half or total_int == 1) else allocation.get(act, 1)
        # n_needed here just sizes the candidate pool (see
        # query_places_for_activity) - actual day-filling is driven by
        # each place's own duration, not a fixed count.
        n_need = max(days_for_act * 6, 12)
        matched = query_places_for_activity(
            user_state    = data["state"],
            predicted_cat = cat,
            top3_cats     = top3,
            n_needed      = n_need,
            exclude_names = used,
        )
        act_pools[act] = matched
        used.update(matched["display_name"].tolist() if len(matched) > 0 else [])

    # ── Build itinerary JSON using the smart duration allocator ────
    itinerary  = []
    all_places = []
    day_budget = 7.0 if is_half else DAY_BUDGET_HOURS

    for day_num in sorted(day_activity_map.keys()):
        day_acts  = day_activity_map[day_num]
        day_label = "HALF DAY" if is_half else f"DAY {day_num}"

        day_places_list = []
        day_used_names = set()

        # Pull from each of today's activity pools, filling the day's
        # time budget by each place's own duration (beach/island and
        # adventure/waterpark places consume more hours than a quick
        # heritage/city stop, so a day's final place count varies).
        for act in day_acts:
            pool = act_pools[act]
            if pool.empty:
                continue
            chosen, remaining_pool, current_hour, remaining_budget = allocate_places_by_duration(pool, day_budget)
            act_pools[act] = remaining_pool  # so tomorrow continues from where today left off

            for p, start_hour, duration in chosen:
                name  = str(p.get("display_name", ""))
                if name in day_used_names:
                    continue
                state = str(p.get("state", data["state"]))
                link, link_type = get_more_info_link(p, name, state)
                place_obj = {
                    "time_slot":     format_time_range(start_hour, duration),
                    "place_name":    name,
                    "category":      str(p.get("category", "")),
                    "duration_hours": duration,
                    "rating":        round(float(p.get("rating_imputed", 0)), 1),
                    "more_info_url": link,
                    "link_type":     link_type,
                    "match_quality": str(p.get("match_quality", "")),
                    "state":         state,
                }
                day_places_list.append(place_obj)
                all_places.append(place_obj)
                day_used_names.add(name)

            # If the day still has worthwhile time left after the main
            # activity places, use it for a food/relax stop rather than
            # squeezing in more sightseeing.
            filler, _ = pick_filler_stop(
                user_state=data["state"],
                current_hour=current_hour,
                remaining_budget=remaining_budget,
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

            day_budget = 7.0 if is_half else DAY_BUDGET_HOURS  # reset for next day

        # ── Accommodation check-in (only if relevant to this user) ──
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

        itinerary.append({
            "day":                day_num,
            "day_label":          day_label,
            "activity":           day_acts[0] if len(day_acts) == 1 else "Mixed",
            "predicted_category": act_preds[day_acts[0]]["category"] if len(day_acts) == 1 else "Mixed",
            "places":             day_places_list,
        })

    # ── Trip summary ──────────────────────────────────────────────
    avg_rating = round(float(np.mean([p["rating"] for p in all_places])), 2) if all_places else 0.0
    day_alloc  = [
        {"activity": act, "days": n, "category": act_preds[act]["category"]}
        for act, n in allocation.items()
    ] if allocation else []

    trip_summary = {
        "state":          data["state"],
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

    # ── Return response ───────────────────────────────────────────
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
