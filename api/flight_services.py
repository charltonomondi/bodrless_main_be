import requests
import os
from dotenv import load_dotenv
import logging
from django.conf import settings

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

TRAVELDUQA_API_URL = "https://www.app.travelduqa.africa/connect/getOffers"
TRAVELDUQA_API_KEY = os.getenv("TRAVELDUQA_API_KEY") or getattr(settings, 'TRAVELDUQA_API_KEY', '')


def format_duration(duration_iso):
    """Format PT1H8M → "1h 8m" """
    hours = minutes = 0
    duration = duration_iso.replace("PT", "")
    if "H" in duration:
        hours_part, duration = duration.split("H")
        hours = int(hours_part)
    if "M" in duration:
        minutes = int(duration.replace("M", ""))
    return f"{hours}h {minutes}m" if hours or minutes else "N/A"


def search_flights(from_location, to_location, date, adult_count=1, child_count=0, infant_count=0, cabin_class="economy"):
    """
    Search for flights using TravelDuqa API
    """
    log.info(f"🔍 Searching flights from {from_location} to {to_location} on {date} for {adult_count} adults, {child_count} children, and {infant_count} infants in {cabin_class} class.")
    log.info(f"TravelDuqa API key configured: {'Yes' if TRAVELDUQA_API_KEY else 'No'}")
    log.info(f"API URL: {TRAVELDUQA_API_URL}")

    if not TRAVELDUQA_API_KEY:
        log.error("❌ TravelDuqa API key not configured")
        return {
            "result": "error",
            "message": "TravelDuqa API key not configured"
        }

    url = TRAVELDUQA_API_URL
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Travelduqa-Version": "v1",
        "Authorization": f"Bearer {TRAVELDUQA_API_KEY}"
    }

    payload = {
        "journey": {
            "flight_type": "oneway",
            "cabin_class": cabin_class,
            "depature": from_location.upper(),
            "arrival": to_location.upper(),
            "depature_date": date,
            "arrival_date": "-",
            "adult_count": adult_count,
            "child_count": child_count,
            "infant_count": infant_count,
            "currency": "KES",
            "page": {
                "length": "10"
            }
        }
    }

    log.info(f"📤 API Request: {url}")
    log.info(f"📤 Headers: {headers}")
    log.info(f"📤 Payload: {payload}")

    try:
        log.info("🚀 Making TravelDuqa API request...")
        max_retries = 2
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                log.info(f"📤 Attempt {attempt + 1}/{max_retries}")
                response = requests.post(url, json=payload, headers=headers, timeout=15)
                log.info(f"📥 API Response status: {response.status_code}")

                if response.status_code == 429:  # Rate limited
                    if attempt < max_retries - 1:
                        log.warning(f"⏳ Rate limited, waiting {retry_delay}s before retry...")
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        log.error("❌ Rate limited and max retries exceeded")
                        return {"result": "error", "message": "API rate limit exceeded"}

                response.raise_for_status()
                data = response.json()
                log.info(f"📥 Response data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                log.info("✅ Flight search response received successfully")
                break  # Success, exit retry loop

            except requests.exceptions.RequestException as retry_error:
                if attempt < max_retries - 1:
                    log.warning(f"⏳ Request failed (attempt {attempt + 1}), retrying: {retry_error}")
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    log.error(f"❌ All retry attempts failed")
                    raise retry_error

        offers = []

        for offer in data.get("data", []):
            # Extract comprehensive flight information
            slices = offer.get("slices", [])
            if not slices:
                continue

            slice_data = slices[0]  # First slice for one-way flights
            segments = slice_data.get("segments", [])

            if not segments:
                continue

            # Handle multiple segments (layovers)
            flight_segments = []
            total_duration = slice_data.get("duration", "PT0M")

            for i, segment in enumerate(segments):
                origin = segment.get("origin", {})
                destination = segment.get("destination", {})
                carrier = segment.get("marketing_carrier", {})
                operating_carrier = segment.get("operating_carrier", carrier)

                # Terminal and Gate information
                departure_terminal = segment.get("departure_terminal", "")
                arrival_terminal = segment.get("arrival_terminal", "")
                departure_gate = segment.get("departure_gate", "")
                arrival_gate = segment.get("arrival_gate", "")

                # Aircraft information
                aircraft = segment.get("aircraft", {})
                aircraft_type = aircraft.get("iata_code", "") if aircraft else ""

                # Flight segment details
                segment_info = {
                    "segment_number": i + 1,
                    "departure_airport": f"{origin.get('city_name', '')} ({origin.get('iata_code', '')})",
                    "arrival_airport": f"{destination.get('city_name', '')} ({destination.get('iata_code', '')})",
                    "departure_time": segment.get("departing_at", ""),
                    "arrival_time": segment.get("arriving_at", ""),
                    "flight_number": f"{carrier.get('iata_code', '')}{segment.get('marketing_carrier_flight_number', '')}",
                    "operating_airline": operating_carrier.get("name", ""),
                    "operating_flight_number": f"{operating_carrier.get('iata_code', '')}{segment.get('operating_carrier_flight_number', '')}",
                    "aircraft_type": aircraft_type,
                    "duration": format_duration(segment.get("duration", "PT0M")),
                    "departure_terminal": departure_terminal,
                    "arrival_terminal": arrival_terminal,
                    "departure_gate": departure_gate,
                    "arrival_gate": arrival_gate,
                    "stops": len(segments) - 1,
                    "is_layover": i < len(segments) - 1
                }

                flight_segments.append(segment_info)

            # Extract fare breakdown
            fare_details = offer.get("fare_details", {})
            passenger_fares = offer.get("passenger_fares", [])

            # Price breakdown
            base_fare = 0
            taxes = 0
            fees = 0
            surcharges = 0

            if passenger_fares:
                for passenger_fare in passenger_fares:
                    fare_components = passenger_fare.get("fare_components", [])
                    for component in fare_components:
                        component_type = component.get("type", "").lower()
                        amount = float(component.get("amount", 0))

                        if "base" in component_type or "fare" in component_type:
                            base_fare += amount
                        elif "tax" in component_type:
                            taxes += amount
                        elif "fee" in component_type or "surcharge" in component_type:
                            fees += amount

            # Baggage information
            baggage_info = []
            if "baggage" in offer:
                baggage_data = offer["baggage"]
                if isinstance(baggage_data, dict):
                    # Checked baggage
                    if "checked" in baggage_data:
                        checked = baggage_data["checked"]
                        if isinstance(checked, dict):
                            baggage_info.append({
                                "type": "checked",
                                "allowance": f"{checked.get('quantity', 0)} pieces, {checked.get('weight', 'N/A')} kg each",
                                "cost": checked.get("cost", "Included")
                            })

                    # Carry-on baggage
                    if "carry_on" in baggage_data:
                        carry_on = baggage_data["carry_on"]
                        if isinstance(carry_on, dict):
                            baggage_info.append({
                                "type": "carry_on",
                                "allowance": f"{carry_on.get('quantity', 0)} pieces, {carry_on.get('weight', 'N/A')} kg each",
                                "cost": carry_on.get("cost", "Included")
                            })

            # Enhanced offer data
            enhanced_offer = {
                "id": offer.get("id", ""),
                "total_amount": offer.get("total_amount", 0),
                "total_currency": offer.get("total_currency", "KES"),
                "cabin_class": slice_data.get("cabin_class_marketing_name", "Economy"),

                # Flight route details
                "flight_segments": flight_segments,
                "total_duration": format_duration(total_duration),
                "total_stops": len(segments) - 1,
                "stops_airports": [seg["arrival_airport"] for seg in flight_segments[:-1]] if len(flight_segments) > 1 else [],

                # Terminal and gate information
                "departure_terminal": flight_segments[0]["departure_terminal"] if flight_segments else "",
                "arrival_terminal": flight_segments[-1]["arrival_terminal"] if flight_segments else "",
                "departure_gate": flight_segments[0]["departure_gate"] if flight_segments else "",
                "arrival_gate": flight_segments[-1]["arrival_gate"] if flight_segments else "",

                # Fare breakdown
                "fare_breakdown": {
                    "base_fare": base_fare,
                    "taxes": taxes,
                    "fees": fees,
                    "surcharges": surcharges,
                    "total": offer.get("total_amount", 0)
                },

                # Baggage information
                "baggage_info": baggage_info,

                # Additional services (if available)
                "available_services": {
                    "seat_selection": offer.get("seat_selection_available", False),
                    "meal_included": offer.get("meal_included", False),
                    "wifi_available": offer.get("wifi_available", False),
                    "entertainment_available": offer.get("entertainment_available", False)
                },

                # Booking and conditions
                "booking_code": offer.get("booking_code", ""),
                "fare_conditions": offer.get("fare_conditions", {}),
                "changeable": offer.get("changeable", False),
                "refundable": offer.get("refundable", False)
            }

            offers.append(enhanced_offer)

        log.info(f"✅ Found {len(offers)} flight offers")
        return {
            "result": "success",
            "offers": offers[:5]  # Return top 5
        }

    except requests.exceptions.RequestException as e:
        log.error(f"❌ Flight search request error: {e}")
        log.error(f"Error type: {type(e).__name__}")
        if hasattr(e, 'response') and e.response is not None:
            log.error(f"Response status: {e.response.status_code}")
            log.error(f"Response body: {e.response.text}")
        return {"result": "error", "message": f"API request failed: {str(e)}"}
    except ValueError as e:
        log.error(f"❌ JSON parsing error: {e}")
        return {"result": "error", "message": f"Invalid API response format: {str(e)}"}
    except Exception as e:
        log.error(f"❌ Flight search unexpected error: {e}")
        log.error(f"Error type: {type(e).__name__}")
        import traceback
        log.error(f"Traceback: {traceback.format_exc()}")
        return {"result": "error", "message": "Internal server error"}