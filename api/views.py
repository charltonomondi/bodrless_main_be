from django.shortcuts import render
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.middleware.csrf import get_token
from django.utils.decorators import method_decorator
from django.views import View
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.authtoken.models import Token
from rest_framework.decorators import api_view, permission_classes
from django.db import transaction
import json
import requests
import os
from datetime import datetime
from .models import UserProfile, FlightSearch, FlightBooking
from .flight_services import search_flights
from django.conf import settings


from mistralai import Mistral
import logging
from django.core.mail import send_mail

# Import IntaSend SDK with error handling
try:
    from intasend import APIService
    INTASEND_AVAILABLE = True
except ImportError as e:
    logger.warning(f"IntaSend SDK not available: {e}")
    APIService = None
    INTASEND_AVAILABLE = False

logger = logging.getLogger(__name__)


def parse_and_validate_price(price_str, field_name='price'):
    """
    Parse and validate a price string, handling various formats like 'KES 15,000' or '$150.50'

    Args:
        price_str (str): The price string to parse
        field_name (str): Name of the field for error messages

    Returns:
        tuple: (amount, currency) where amount is float and currency is str

    Raises:
        ValueError: If price format is invalid or amount is not positive
    """
    if not price_str or not isinstance(price_str, str):
        raise ValueError(f'{field_name} is required and must be a valid string')

    # Remove all whitespace
    price_str = price_str.strip()

    if not price_str:
        raise ValueError(f'{field_name} cannot be empty')

    # Extract currency and amount
    currency_symbols = ['KES', 'USD', '$']
    currency = 'USD'  # default

    # Check for currency prefix
    for symbol in currency_symbols:
        if price_str.startswith(symbol):
            if symbol == '$':
                currency = 'USD'
            elif symbol == 'KES':
                currency = 'KES'
            else:
                currency = symbol
            price_str = price_str[len(symbol):].strip()
            break

    # Handle currency at the end (e.g., "150 KES")
    if price_str.upper().endswith(' KES'):
        currency = 'KES'
        price_str = price_str[:-4].strip()
    elif price_str.upper().endswith(' USD'):
        currency = 'USD'
        price_str = price_str[:-4].strip()

    # Remove any remaining currency symbols and commas
    price_str = price_str.replace('$', '').replace('KES', '').replace(',', '').strip()

    try:
        amount = float(price_str)

        if amount <= 0:
            raise ValueError(f'{field_name} must be greater than 0')

        if amount > 1000000:  # Reasonable upper limit
            logger.warning(f'Very high price detected: {amount} {currency}')

        return amount, currency

    except ValueError as e:
        if 'could not convert string to float' in str(e):
            raise ValueError(f'Invalid {field_name} format: "{price_str}". Expected format like "KES 15,000" or "$150.50"')
        else:
            raise ValueError(f'Invalid {field_name}: {str(e)}')


@method_decorator(csrf_exempt, name='dispatch')
class RegisterView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        try:
            data = json.loads(request.body)
            username = data.get('username')
            email = data.get('email')
            password = data.get('password')
            first_name = data.get('first_name', '')
            last_name = data.get('last_name', '')

            if User.objects.filter(username=username).exists():
                return Response({'error': 'Username already exists'}, status=400)

            if User.objects.filter(email=email).exists():
                return Response({'error': 'Email already exists'}, status=400)

            user = User.objects.create_user(
                username=username,
                email=email,
                password=password,
                first_name=first_name,
                last_name=last_name
            )


            # Create user profile
            UserProfile.objects.create(user=user)

            # Send welcome email
            try:
                send_mail(
                    subject='Welcome to Bodrless!',
                    message=f"Hi {first_name or username},\n\nThank you for signing up to Bodrless. Your account has been created successfully!\n\nHappy travels!\nThe Bodrless Team",
                    from_email=None,
                    recipient_list=[email],
                    fail_silently=True,
                )
            except Exception as e:
                logger.warning(f"Failed to send welcome email: {e}")

            # Create token
            token, created = Token.objects.get_or_create(user=user)

            return Response({
                'message': 'User created successfully',
                'token': token.key,
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'first_name': user.first_name,
                    'last_name': user.last_name,
                }
            })

        except Exception as e:
            logger.error(f"Registration error: {e}")
            return Response({'error': 'Registration failed'}, status=400)


@method_decorator(csrf_exempt, name='dispatch')
class LoginView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        try:
            logger.info(f"🔐 Login attempt from {request.META.get('REMOTE_ADDR', 'unknown')}")
            logger.info(f"Request headers: {dict(request.headers)}")

            # Handle both JSON and form data
            if request.content_type == 'application/x-www-form-urlencoded':
                data = {
                    'username': request.POST.get('username'),
                    'password': request.POST.get('password')
                }
            else:
                data = json.loads(request.body)

            username = data.get('username')
            password = data.get('password')

            logger.info(f"Login credentials received for user: {username}")

            if not username or not password:
                logger.warning("Missing username or password")
                return Response({
                    'error': 'Username and password are required',
                    'details': 'Please provide both username and password'
                }, status=400)

            user = authenticate(username=username, password=password)

            if user is not None:
                logger.info(f"✅ Authentication successful for user: {username}")
                token, created = Token.objects.get_or_create(user=user)

                logger.info(f"🔑 Token {'created' if created else 'retrieved'} for user: {username}")

                return Response({
                    'message': 'Login successful',
                    'token': token.key,
                    'user': {
                        'id': user.id,
                        'username': user.username,
                        'email': user.email,
                        'first_name': user.first_name,
                        'last_name': user.last_name,
                    }
                })
            else:
                logger.warning(f"❌ Authentication failed for user: {username}")
                return Response({
                    'error': 'Invalid credentials',
                    'details': 'The username or password you entered is incorrect'
                }, status=401)

        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON decode error: {e}")
            return Response({
                'error': 'Invalid JSON format',
                'details': 'Please ensure your request contains valid JSON data'
            }, status=400)
        except Exception as e:
            logger.error(f"❌ Login error: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return Response({
                'error': 'Login failed',
                'details': 'An unexpected error occurred during login'
            }, status=500)


@method_decorator(csrf_exempt, name='dispatch')
class LogoutView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            request.user.auth_token.delete()
            return Response({'message': 'Logout successful'})
        except Exception as e:
            logger.error(f"Logout error: {e}")
            return Response({'error': 'Logout failed'}, status=400)


@method_decorator(csrf_exempt, name='dispatch')
class ProfileView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            profile = UserProfile.objects.get(user=request.user)
            return Response({
                'user': {
                    'id': request.user.id,
                    'username': request.user.username,
                    'email': request.user.email,
                    'first_name': request.user.first_name,
                    'last_name': request.user.last_name,
                },
                'profile': {
                    'phone_number': profile.phone_number,
                    'date_of_birth': profile.date_of_birth,
                    'passport_number': profile.passport_number,
                    'passport_expiry': profile.passport_expiry,
                    'preferred_currency': profile.preferred_currency,
                }
            })
        except UserProfile.DoesNotExist:
            return Response({'error': 'Profile not found'}, status=404)

    def put(self, request):
        try:
            data = json.loads(request.body)
            profile = UserProfile.objects.get(user=request.user)

            # Update user fields
            user_fields = ['first_name', 'last_name', 'email']
            for field in user_fields:
                if field in data:
                    setattr(request.user, field, data[field])
            request.user.save()

            # Update profile fields
            profile_fields = ['phone_number', 'date_of_birth', 'passport_number', 'passport_expiry', 'preferred_currency']
            for field in profile_fields:
                if field in data:
                    setattr(profile, field, data[field])
            profile.save()

            return Response({'message': 'Profile updated successfully'})

        except Exception as e:
            logger.error(f"Profile update error: {e}")
            return Response({'error': 'Profile update failed'}, status=400)


@method_decorator(csrf_exempt, name='dispatch')
class ChatView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        try:
            logger.info("🎯 CHAT ENDPOINT HIT - Starting request processing")
            logger.info(f"Request method: {request.method}")
            logger.info(f"Request path: {request.path}")
            logger.info(f"Remote address: {request.META.get('REMOTE_ADDR')}")
            logger.info(f"User agent: {request.META.get('HTTP_USER_AGENT')}")

            # Initialize Mistral client at request time to ensure env is loaded
            client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

            logger.info(f"🚀 Chat request received. Method: {request.method}")
            logger.info(f"Request headers: {dict(request.headers)}")
            logger.info(f"Request body: {request.body.decode('utf-8') if request.body else 'No body'}")

            data = json.loads(request.body)
            user_input = data.get('user_input', '')
            logger.info(f"📝 User input: {user_input}")

            if not user_input.strip():
                logger.warning("Empty user input received")
                return Response({'reply': 'Please provide a valid message.'})

            # Check if user is authenticated
            is_authenticated = request.user.is_authenticated
            logger.info(f"User authenticated: {is_authenticated}")

            # Log if user input contains flight-only keywords
            flight_only_keywords = [
                "only flight", "flight only", "just flight", "flights only", "only flights", "just flights",
                "only need flight", "need only flight", "just need flight", "only want flight", "want only flight",
                "just want flight", "only the flight", "just the flight", "only this flight", "just this flight",
                "no thanks", "no additional", "no other services", "no extras", "no more", "that's all",
                "nothing else", "just that", "only that", "no", "nope"
            ]
            user_input_lower = user_input.lower()
            has_flight_only = any(keyword in user_input_lower for keyword in flight_only_keywords)
            logger.info(f"User input contains flight-only keywords: {has_flight_only}")

            # System prompt for travel agent - different for authenticated vs non-authenticated users
            if is_authenticated:
                system_prompt = """You are a helpful assistant for a travel platform.

                When users ask about flights or travel between locations, immediately show available flight options using this format:

                TRAVEL_OPTIONS:
                [
                  {
                    "title": "Direct Flight NBO-MBA",
                    "type": "Flight",
                    "price": "$85",
                    "duration": "1h 15m",
                    "rating": "4.5 stars",
                    "highlights": ["Direct flight", "No layovers", "On-time service"],
                    "description": "Quick direct flight from Nairobi to Mombasa"
                  },
                  {
                    "title": "Economy Flight NBO-MBA",
                    "type": "Flight",
                    "price": "$65",
                    "duration": "1h 20m",
                    "rating": "4.2 stars",
                    "highlights": ["Budget option", "Reliable service", "Comfortable seating"],
                    "description": "Affordable economy flight with good amenities"
                  }
                ]
                END_TRAVEL_OPTIONS

                Always be friendly and helpful."""
            else:
                system_prompt = """You are a helpful assistant for a travel platform.

                When users ask about flights or travel between locations, immediately show available flight options using this format:

                TRAVEL_OPTIONS:
                [
                  {
                    "title": "Direct Flight NBO-MBA",
                    "type": "Flight",
                    "price": "$85",
                    "duration": "1h 15m",
                    "rating": "4.5 stars",
                    "highlights": ["Direct flight", "No layovers", "On-time service"],
                    "description": "Quick direct flight from Nairobi to Mombasa"
                  },
                  {
                    "title": "Economy Flight NBO-MBA",
                    "type": "Flight",
                    "price": "$65",
                    "duration": "1h 20m",
                    "rating": "4.2 stars",
                    "highlights": ["Budget option", "Reliable service", "Comfortable seating"],
                    "description": "Affordable economy flight with good amenities"
                  }
                ]
                END_TRAVEL_OPTIONS

                Always be friendly and helpful. For booking, suggest users create an account."""

            # Enhanced flight query detection with more cities and destinations
            # Separate city keywords from general flight keywords to avoid false positives
            city_keywords = [
                'nairobi', 'nbo', 'mombasa', 'mba', 'kenya', 'toronto', 'canada',
                'london', 'uk', 'dubai', 'uae', 'paris', 'france', 'new york', 'nyc',
                'los angeles', 'la', 'tokyo', 'japan', 'sydney', 'australia', 'cape town',
                'johannesburg', 'jnb', 'cairo', 'egypt', 'doha', 'qatar', 'amsterdam',
                'netherlands', 'singapore', 'bangkok', 'thailand', 'istanbul', 'turkey',
                'mumbai', 'india', 'delhi', 'beijing', 'china', 'hong kong', 'seoul',
                'korea', 'moscow', 'russia', 'berlin', 'germany', 'rome', 'italy',
                'madrid', 'spain', 'vienna', 'austria', 'stockholm', 'sweden',
                'copenhagen', 'denmark', 'oslo', 'norway', 'helsinki', 'finland',
                'warsaw', 'poland', 'prague', 'czech', 'budapest', 'hungary',
                'bucharest', 'romania', 'sofia', 'bulgaria', 'athens', 'greece',
                'lisbon', 'portugal', 'dublin', 'ireland', 'edinburgh', 'scotland',
                'cardiff', 'wales', 'belfast', 'northern ireland', 'zurich', 'switzerland',
                'geneva', 'brussels', 'belgium', 'luxembourg', 'monaco', 'san marino',
                'vatican', 'andorra', 'liechtenstein', 'malta', 'cyprus', 'iceland',
                'reykjavik', 'bergen', 'gothenburg', 'malmö', 'aarhus', 'trondheim',
                'stavanger', 'espoo', 'tampere', 'vantaa', 'turku', 'oulu', 'krakow',
                'lodz', 'wroclaw', 'poznan', 'gdansk', 'szczecin', 'bydgoszcz',
                'lublin', 'katowice', 'bialystok', 'gdynia', 'czestochowa', 'radom',
                'torun', 'kielce', 'gliwice', 'zabrze', 'bytom', 'zielona gora',
                'rybnik', 'rzeszow', 'tychy', 'dabrowa gornicza', 'pleszew', 'elk',
                'brno', 'ostrava', 'plzen', 'liberec', 'olomouc', 'ceske budejovice',
                'hradec kralove', 'usti nad labem', 'pardubice', 'havirov', 'kladno',
                'most', 'opava', 'frydek-mistek', 'karvina', 'jicin', 'teplice', 'decin',
                'chomutov', 'jihlava', 'prostějov', 'prerov', 'pribram', 'kolín',
                'kromeriz', 'skutec', 'litomerice', 'nachod', 'beroun', 'mlada boleslav'
            ]

            # General flight-related keywords (excluding city names to avoid false positives)
            flight_keywords = [
                'flight', 'flights', 'fly', 'flying', 'travel', 'trip', 'book', 'booking',
                'destination', 'departure', 'airline', 'airport', 'plane', 'ticket', 'tickets'
            ]

            # Train-related keywords
            train_keywords = [
                'train', 'trains', 'rail', 'railway', 'railroad', 'locomotive', 'coach', 'carriage',
                'station', 'track', 'rail travel', 'train journey', 'train ticket', 'train booking'
            ]

            user_input_lower = user_input.lower()
            is_flight_query = any(keyword in user_input_lower for keyword in flight_keywords)
            is_train_query = any(keyword in user_input_lower for keyword in train_keywords)

            if is_train_query:
                logger.info("Train query detected, generating train options")

                # Generate train options for Nairobi to Mombasa route
                travel_options = []

                # First Class train option
                travel_options.append({
                    "title": "First Class Train NBO-MBA",
                    "type": "Train",
                    "price": "KES 4,700",
                    "duration": "5h 30m",
                    "rating": "4.8 stars",
                    "highlights": [
                        "Reclining, padded seats with more legroom",
                        "Air-conditioned coaches",
                        "Complimentary snack or light meal",
                        "On-board power outlets",
                        "Wi-Fi (when available)",
                        "30 kg baggage allowance",
                        "Priority boarding and alighting"
                    ],
                    "description": "Premium first class train service from Nairobi to Mombasa",
                    "detailed_info": {
                        "route": "NBO → MBA",
                        "train_type": "Express Passenger",
                        "class": "First Class",
                        "departure_time": "07:00",
                        "arrival_time": "12:30",
                        "train_number": "TR001",
                        "operating_company": "Kenya Railways",
                        "coaches": "Air-conditioned",
                        "amenities": ["Reclining seats", "Power outlets", "Wi-Fi", "Complimentary meal", "Priority service"],
                        "baggage_allowance": "30 kg",
                        "stops": ["Limited stops", "Direct service available"],
                        "services": {
                            "food_service": True,
                            "wifi_available": True,
                            "power_outlets": True,
                            "air_conditioning": True,
                            "priority_boarding": True
                        }
                    }
                })

                # Economy Class train option
                travel_options.append({
                    "title": "Economy Class Train NBO-MBA",
                    "type": "Train",
                    "price": "KES 1,700",
                    "duration": "6h 15m",
                    "rating": "4.2 stars",
                    "highlights": [
                        "Standard cushioned seats",
                        "Air-conditioned coaches",
                        "20 kg baggage allowance",
                        "Access to on-board vendor for snacks and drinks (purchase required)"
                    ],
                    "description": "Affordable economy class train service from Nairobi to Mombasa",
                    "detailed_info": {
                        "route": "NBO → MBA",
                        "train_type": "Standard Passenger",
                        "class": "Economy Class",
                        "departure_time": "14:30",
                        "arrival_time": "20:45",
                        "train_number": "TR003",
                        "operating_company": "Kenya Railways",
                        "coaches": "Air-conditioned",
                        "amenities": ["Comfortable seating", "Air conditioning", "On-board vendor"],
                        "baggage_allowance": "20 kg",
                        "stops": ["Regular stops", "Scenic route"],
                        "services": {
                            "food_service": False,
                            "wifi_available": False,
                            "power_outlets": False,
                            "air_conditioning": True,
                            "priority_boarding": False
                        }
                    }
                })

                reply = "Here are your train options from Nairobi to Mombasa. Click on any option to proceed with booking!"

                return Response({
                    'reply': reply,
                    'is_authenticated': is_authenticated,
                    'travel_options': travel_options,
                    'payment_options': None,
                    'booking_flow': {
                        'enabled': True,
                        'steps': [
                            {
                                'step': 'train_selection',
                                'message': 'Select a train service to proceed with booking',
                                'action': 'select_train'
                            },
                            {
                                'step': 'passenger_details',
                                'message': 'Enter passenger information',
                                'action': 'enter_passenger_details'
                            },
                            {
                                'step': 'payment',
                                'message': 'Choose payment method',
                                'action': 'show_payment_options'
                            }
                        ]
                    }
                })

            elif is_flight_query:
                logger.info("Flight query detected, attempting to fetch real flight data")

                # Extract cities from the query for dynamic responses
                cities = []

                # First, try to extract from common patterns like "from X to Y"
                if 'from' in user_input_lower and 'to' in user_input_lower:
                    # Use regex to find "from X to Y" pattern
                    import re
                    pattern = r'from\s+(.+?)\s+to\s+(.+)'
                    match = re.search(pattern, user_input_lower)

                    if match:
                        from_part = match.group(1).strip()
                        to_part = match.group(2).strip()

                        if from_part:
                            cities.append(from_part.title())
                        if to_part:
                            cities.append(to_part.title())

                # If pattern matching didn't find cities, look for individual city names
                if not cities:
                    words = user_input_lower.split()
                    for word in words:
                        if word in city_keywords and len(word) > 2:
                            cities.append(word.title())

                # If still no cities found, try to find multi-word cities
                if not cities:
                    words = user_input_lower.split()
                    # Look for "new york" pattern
                    if 'new york' in user_input_lower:
                        cities.append('New York')
                    # Look for other potential multi-word cities
                    for i in range(len(words) - 1):
                        two_word_city = f"{words[i]} {words[i+1]}"
                        if two_word_city in city_keywords:
                            cities.append(two_word_city.title())

                # Default cities if none detected
                if not cities:
                    cities = ['Nairobi', 'Toronto']

                from_city = cities[0] if len(cities) > 0 else 'Nairobi'
                to_city = cities[1] if len(cities) > 1 else 'Toronto'

                # Try to get real flight data from TravelDuqa API
                try:
                    # Map city names to airport codes
                    city_to_code = {
                        'Nairobi': 'NBO', 'Mombasa': 'MBA', 'Toronto': 'YYZ', 'London': 'LHR',
                        'Dubai': 'DXB', 'Paris': 'CDG', 'New York': 'JFK', 'Tokyo': 'NRT',
                        'Los Angeles': 'LAX', 'Sydney': 'SYD', 'Cairo': 'CAI', 'Doha': 'DOH',
                        'Amsterdam': 'AMS', 'Singapore': 'SIN', 'Bangkok': 'BKK', 'Istanbul': 'IST',
                        'Mumbai': 'BOM', 'Delhi': 'DEL', 'Beijing': 'PEK', 'Hong Kong': 'HKG',
                        'Seoul': 'ICN', 'Moscow': 'SVO', 'Berlin': 'BER', 'Rome': 'FCO',
                        'Madrid': 'MAD', 'Vienna': 'VIE', 'Stockholm': 'ARN', 'Copenhagen': 'CPH',
                        'Oslo': 'OSL', 'Helsinki': 'HEL', 'Warsaw': 'WAW', 'Prague': 'PRG',
                        'Budapest': 'BUD', 'Zurich': 'ZRH', 'Geneva': 'GVA', 'Brussels': 'BRU',
                        'Ethiopia': 'ADD', 'Addis Ababa': 'ADD', 'Canada': 'YYZ', 'Montreal': 'YUL'
                    }

                    from_code = city_to_code.get(from_city, from_city[:3].upper())
                    to_code = city_to_code.get(to_city, to_city[:3].upper())

                    # Get today's date for the search
                    from datetime import datetime
                    today = datetime.now().strftime('%Y-%m-%d')

                    logger.info(f"🔍 Searching flights from {from_city} ({from_code}) to {to_city} ({to_code}) on {today}")
                    logger.info(f"TravelDuqa API key configured: {'Yes' if os.getenv('TRAVELDUQA_API_KEY') else 'No'}")

                    # Call the TravelDuqa API
                    flight_results = search_flights(
                        from_location=from_code,
                        to_location=to_code,
                        date=today,
                        adult_count=1,
                        child_count=0,
                        infant_count=0,
                        cabin_class='economy'
                    )

                    logger.info(f"TravelDuqa API response: {flight_results.get('result', 'unknown')}")
                    if flight_results.get('result') == 'error':
                        logger.error(f"TravelDuqa API error: {flight_results.get('message', 'Unknown error')}")

                    if flight_results.get('result') == 'success' and flight_results.get('offers'):
                        # Format real flight data with enhanced details
                        travel_options = []
                        for offer in flight_results['offers'][:5]:  # Limit to 5 results
                            # Extract first and last segments for route display
                            segments = offer.get('flight_segments', [])
                            if segments:
                                first_segment = segments[0]
                                last_segment = segments[-1]

                                # Create route string (e.g., "NBO → MBA" or "NBO → DXB → JFK")
                                route_parts = [seg['departure_airport'].split(' (')[1].split(')')[0] for seg in segments]
                                route_parts.append(last_segment['arrival_airport'].split(' (')[1].split(')')[0])
                                route_string = " → ".join(route_parts)

                                # Create flight numbers string
                                flight_numbers = [seg['flight_number'] for seg in segments]
                                flight_numbers_str = " / ".join(flight_numbers)

                                # Terminal and gate info
                                terminal_info = []
                                if first_segment.get('departure_terminal'):
                                    terminal_info.append(f"Departs from Terminal {first_segment['departure_terminal']}")
                                if last_segment.get('arrival_terminal'):
                                    terminal_info.append(f"Arrives at Terminal {last_segment['arrival_terminal']}")

                                # Stops information
                                stops_info = []
                                if offer.get('total_stops', 0) > 0:
                                    stops_info.append(f"{offer['total_stops']} stop(s)")
                                    if offer.get('stops_airports'):
                                        stops_info.append(f"via {', '.join(offer['stops_airports'])}")

                                # Price breakdown
                                fare_breakdown = offer.get('fare_breakdown', {})
                                price_details = []
                                if fare_breakdown.get('base_fare'):
                                    price_details.append(f"Base: {offer['total_currency']} {fare_breakdown['base_fare']}")
                                if fare_breakdown.get('taxes'):
                                    price_details.append(f"Taxes: {offer['total_currency']} {fare_breakdown['taxes']}")

                                # Baggage info
                                baggage_info = offer.get('baggage_info', [])
                                baggage_details = []
                                for baggage in baggage_info:
                                    if baggage['type'] == 'checked':
                                        baggage_details.append(f"Checked: {baggage['allowance']}")
                                    elif baggage['type'] == 'carry_on':
                                        baggage_details.append(f"Carry-on: {baggage['allowance']}")

                                # Create highlights
                                highlights = ["Real flight data", "Live pricing", "Actual availability"]
                                if offer.get('total_stops', 0) == 0:
                                    highlights.append("Direct flight")
                                else:
                                    highlights.append(f"{offer['total_stops']} stop(s)")
                                if offer.get('changeable'):
                                    highlights.append("Changeable ticket")
                                if offer.get('refundable'):
                                    highlights.append("Refundable ticket")

                                travel_options.append({
                                    "title": f"{first_segment['flight_number']} - {route_string}",
                                    "type": "Flight",
                                    "price": f"{offer['total_currency']} {offer['total_amount']}",
                                    "duration": offer['total_duration'],
                                    "rating": "4.5 stars",
                                    "highlights": highlights,
                                    "description": f"{first_segment.get('operating_airline', offer.get('airline', 'Flight'))} {flight_numbers_str} from {first_segment['departure_airport']} to {last_segment['arrival_airport']}",
                                    "detailed_info": {
                                        "route": route_string,
                                        "flight_segments": segments,
                                        "terminal_info": terminal_info,
                                        "stops_info": stops_info,
                                        "price_breakdown": price_details,
                                        "baggage_info": baggage_details,
                                        "fare_breakdown": fare_breakdown,
                                        "services": offer.get('available_services', {}),
                                        "booking_info": {
                                            "changeable": offer.get('changeable', False),
                                            "refundable": offer.get('refundable', False)
                                        }
                                    }
                                })
                            else:
                                # Fallback for old format
                                travel_options.append({
                                    "title": f"{offer.get('airline', 'Flight')} {offer.get('flight_number', '')}",
                                    "type": "Flight",
                                    "price": offer.get('price', 'N/A'),
                                    "duration": offer.get('duration', 'N/A'),
                                    "rating": "4.5 stars",
                                    "highlights": ["Real flight data", "Live pricing", "Actual availability"],
                                    "description": f"Flight from {offer.get('origin', 'N/A')} to {offer.get('destination', 'N/A')}"
                                })

                        reply = f"Here are your real flight options from {from_city} to {to_city}. Click on any option to proceed with booking!"

                        return Response({
                            'reply': reply,
                            'is_authenticated': is_authenticated,
                            'travel_options': travel_options,
                            'payment_options': None,
                            'booking_flow': {
                                'enabled': True,
                                'steps': [
                                    {
                                        'step': 'flight_selection',
                                        'message': 'Select a flight to proceed with booking',
                                        'action': 'select_flight'
                                    },
                                    {
                                        'step': 'passenger_details',
                                        'message': 'Enter passenger information',
                                        'action': 'enter_passenger_details'
                                    },
                                    {
                                        'step': 'payment',
                                        'message': 'Choose payment method',
                                        'action': 'show_payment_options'
                                    }
                                ]
                            }
                        })
                    else:
                        logger.warning(f"TravelDuqa API call failed or returned no flights: {flight_results.get('message', 'Unknown error')}")
                        logger.info(f"No flights found for route {from_code} to {to_code}")

                        # Try alternative routes if the direct route has no flights
                        alternative_routes = []
                        if from_code == 'NBO' and to_code == 'TOR':
                            # Try NBO to LHR (London) as alternative
                            alternative_routes.append(('NBO', 'LHR', 'London'))
                        elif from_code == 'NBO' and to_code == 'JFK':
                            # Try NBO to LHR for New York
                            alternative_routes.append(('NBO', 'LHR', 'London'))

                        for alt_from, alt_to, alt_city in alternative_routes:
                            try:
                                logger.info(f"Trying alternative route: {alt_from} to {alt_to}")
                                alt_results = search_flights(
                                    from_location=alt_from,
                                    to_location=alt_to,
                                    date=today,
                                    adult_count=1,
                                    child_count=0,
                                    infant_count=0,
                                    cabin_class='economy'
                                )

                                if alt_results.get('result') == 'success' and alt_results.get('offers'):
                                    travel_options = []
                                    for offer in alt_results['offers'][:3]:  # Limit to 3 results for alternatives
                                        # Extract first and last segments for route display
                                        segments = offer.get('flight_segments', [])
                                        if segments:
                                            first_segment = segments[0]
                                            last_segment = segments[-1]

                                            # Create route string
                                            route_parts = [seg['departure_airport'].split(' (')[1].split(')')[0] for seg in segments]
                                            route_parts.append(last_segment['arrival_airport'].split(' (')[1].split(')')[0])
                                            route_string = " → ".join(route_parts)

                                            # Create flight numbers string
                                            flight_numbers = [seg['flight_number'] for seg in segments]
                                            flight_numbers_str = " / ".join(flight_numbers)

                                            travel_options.append({
                                                "title": f"{first_segment['flight_number']} - {route_string}",
                                                "type": "Flight",
                                                "price": f"{offer['total_currency']} {offer['total_amount']}",
                                                "duration": offer['total_duration'],
                                                "rating": "4.2 stars",
                                                "highlights": ["Alternative route", "Real flight data", "Live pricing", f"Via {alt_city}"],
                                                "description": f"{first_segment.get('operating_airline', 'Flight')} {flight_numbers_str} from {first_segment['departure_airport']} to {last_segment['arrival_airport']} (Alternative route via {alt_city})"
                                            })
                                        else:
                                            # Fallback for old format
                                            travel_options.append({
                                                "title": f"{offer.get('airline', 'Flight')} {offer.get('flight_number', '')}",
                                                "type": "Flight",
                                                "price": offer.get('price', 'N/A'),
                                                "duration": offer.get('duration', 'N/A'),
                                                "rating": "4.2 stars",
                                                "highlights": ["Alternative route", "Real flight data", "Live pricing"],
                                                "description": f"Flight from {offer.get('origin', 'N/A')} to {offer.get('destination', 'N/A')} (Alternative route via {alt_city})"
                                            })

                                    reply = f"No direct flights found from {from_city} to {to_city}. Here are alternative flight options via {alt_city}:"

                                    return Response({
                                        'reply': reply,
                                        'is_authenticated': is_authenticated,
                                        'travel_options': travel_options,
                                        'payment_options': None
                                    })
                            except Exception as alt_error:
                                logger.warning(f"Alternative route search failed: {alt_error}")
                                continue

                except Exception as api_error:
                    logger.error(f"Error calling TravelDuqa API: {api_error}")

                # Fallback to dynamic options if API fails
                logger.info(f"🔄 Falling back to dynamic flight options for {from_city} to {to_city}")

                # Generate dynamic flight options based on actual destinations
                travel_options = []

                # Define flight data based on routes
                route_data = {
                    ('NBO', 'MBA'): {
                        'distance': 'short',
                        'base_price': 6500,
                        'duration': '1h 15m',
                        'airline': 'Kenya Airways',
                        'flight_number': 'KQ001',
                        'aircraft': 'B737'
                    },
                    ('NBO', 'ADD'): {
                        'distance': 'medium',
                        'base_price': 25000,
                        'duration': '3h 45m',
                        'airline': 'Ethiopian Airlines',
                        'flight_number': 'ET308',
                        'aircraft': 'B787'
                    },
                    ('NBO', 'YYZ'): {
                        'distance': 'long',
                        'base_price': 85000,
                        'duration': '18h 30m',
                        'airline': 'Kenya Airways',
                        'flight_number': 'KQ100',
                        'aircraft': 'B777',
                        'layover': 'AMS'
                    },
                    ('NBO', 'JFK'): {
                        'distance': 'long',
                        'base_price': 75000,
                        'duration': '16h 45m',
                        'airline': 'Kenya Airways',
                        'flight_number': 'KQ002',
                        'aircraft': 'B777',
                        'layover': 'CDG'
                    }
                }

                # Get route info or use defaults
                route_key = (from_code, to_code)
                route_info = route_data.get(route_key, {
                    'distance': 'medium',
                    'base_price': 15000,
                    'duration': '2h 30m',
                    'airline': 'Regional Airlines',
                    'flight_number': 'RA101',
                    'aircraft': 'B737'
                })

                # Create dynamic pricing based on route
                if route_info['distance'] == 'short':
                    prices = [route_info['base_price'], route_info['base_price'] - 1500]
                    durations = [route_info['duration'], '1h 25m']
                elif route_info['distance'] == 'medium':
                    prices = [route_info['base_price'], route_info['base_price'] + 5000]
                    durations = [route_info['duration'], '3h 15m']
                else:  # long haul
                    prices = [route_info['base_price'], route_info['base_price'] + 15000]
                    durations = [route_info['duration'], '20h 15m']

                # Create first flight option (Premium/Direct)
                travel_options.append({
                    "title": f"Direct Flight {from_code} → {to_code}",
                    "type": "Flight",
                    "price": f"KES {prices[0]:,}",
                    "duration": durations[0],
                    "rating": "4.5 stars",
                    "highlights": ["Direct flight", "No layovers", "Premium service", f"Terminal 1A → Terminal {chr(64 + (hash(to_code) % 4) + 1)}"],
                    "description": f"Direct flight from {from_city} ({from_code}) to {to_city} ({to_code})",
                    "detailed_info": {
                        "route": f"{from_code} → {to_code}",
                        "flight_segments": [{
                            "segment_number": 1,
                            "departure_airport": f"{from_city} ({from_code})",
                            "arrival_airport": f"{to_city} ({to_code})",
                            "departure_time": "06:00",
                            "arrival_time": "07:15" if route_info['distance'] == 'short' else "09:45",
                            "flight_number": route_info['flight_number'],
                            "operating_airline": route_info['airline'],
                            "aircraft_type": route_info['aircraft'],
                            "duration": durations[0],
                            "departure_terminal": "1A",
                            "arrival_terminal": chr(64 + (hash(to_code) % 4) + 1),
                            "stops": 0,
                            "is_layover": False
                        }],
                        "total_duration": durations[0],
                        "total_stops": 0,
                        "stops_airports": [],
                        "terminal_info": [f"Departs from Terminal 1A", f"Arrives at Terminal {chr(64 + (hash(to_code) % 4) + 1)}"],
                        "stops_info": ["Direct flight"],
                        "price_breakdown": [f"Base: KES {prices[0] - 1000:,}", f"Taxes: KES 1000"],
                        "baggage_info": ["Checked: 2 pieces, 23 kg each", "Carry-on: 1 piece, 7 kg"],
                        "fare_breakdown": {
                            "base_fare": prices[0] - 1000,
                            "taxes": 1000,
                            "fees": 0,
                            "total": prices[0]
                        },
                        "services": {
                            "seat_selection": True,
                            "meal_included": route_info['distance'] != 'short',
                            "wifi_available": route_info['distance'] == 'long',
                            "entertainment_available": True
                        },
                        "booking_info": {
                            "changeable": True,
                            "refundable": route_info['distance'] != 'short'
                        }
                    }
                })

                # Create second flight option (Economy)
                travel_options.append({
                    "title": f"Economy Flight {from_code} → {to_code}",
                    "type": "Flight",
                    "price": f"KES {prices[1]:,}",
                    "duration": durations[1],
                    "rating": "4.2 stars",
                    "highlights": ["Budget option", "Reliable service", "Comfortable seating", f"Terminal 1B → Terminal {chr(66 + (hash(to_code) % 3))}"],
                    "description": f"Affordable economy flight from {from_city} ({from_code}) to {to_city} ({to_code})",
                    "detailed_info": {
                        "route": f"{from_code} → {to_code}",
                        "flight_segments": [{
                            "segment_number": 1,
                            "departure_airport": f"{from_city} ({from_code})",
                            "arrival_airport": f"{to_city} ({to_code})",
                            "departure_time": "14:30",
                            "arrival_time": "15:55" if route_info['distance'] == 'short' else "17:45",
                            "flight_number": f"{route_info['flight_number'].replace('001', '003')}",
                            "operating_airline": route_info['airline'],
                            "aircraft_type": route_info['aircraft'],
                            "duration": durations[1],
                            "departure_terminal": "1B",
                            "arrival_terminal": chr(66 + (hash(to_code) % 3)),
                            "stops": 0,
                            "is_layover": False
                        }],
                        "total_duration": durations[1],
                        "total_stops": 0,
                        "stops_airports": [],
                        "terminal_info": [f"Departs from Terminal 1B", f"Arrives at Terminal {chr(66 + (hash(to_code) % 3))}"],
                        "stops_info": ["Direct flight"],
                        "price_breakdown": [f"Base: KES {prices[1] - 700:,}", f"Taxes: KES 700"],
                        "baggage_info": ["Checked: 1 piece, 23 kg", "Carry-on: 1 piece, 7 kg"],
                        "fare_breakdown": {
                            "base_fare": prices[1] - 700,
                            "taxes": 700,
                            "fees": 0,
                            "total": prices[1]
                        },
                        "services": {
                            "seat_selection": False,
                            "meal_included": False,
                            "wifi_available": False,
                            "entertainment_available": route_info['distance'] == 'long'
                        },
                        "booking_info": {
                            "changeable": False,
                            "refundable": False
                        }
                    }
                })

                logger.info(f"✅ Generated {len(travel_options)} dynamic flight options for {from_city} to {to_city}")

                reply = f"Here are your flight options from {from_city} to {to_city}. Click on any option to proceed with booking!"

                return Response({
                    'reply': reply,
                    'is_authenticated': is_authenticated,
                    'travel_options': travel_options,
                    'payment_options': None,
                    'booking_flow': {
                        'enabled': True,
                        'steps': [
                            {
                                'step': 'flight_selection',
                                'message': 'Select a flight to proceed with booking',
                                'action': 'select_flight'
                            },
                            {
                                'step': 'passenger_details',
                                'message': 'Enter passenger information',
                                'action': 'enter_passenger_details'
                            },
                            {
                                'step': 'payment',
                                'message': 'Choose payment method',
                                'action': 'show_payment_options'
                            }
                        ]
                    }
                })

            # For non-flight queries, try AI API
            logger.info(f"Attempting to call Mistral API for user input: {user_input[:100]}...")
            try:
                logger.info("Trying mistral-large-latest model...")
                response = client.chat.complete(
                    model="mistral-large-latest",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    max_tokens=1000,
                    temperature=0.7,
                    timeout_ms=30000  # Add 30 second timeout
                )
                logger.info("mistral-large-latest response received successfully")
            except Exception as api_error:
                logger.warning(f"mistral-large-latest failed with error: {str(api_error)}")
                logger.info("Trying mistral-medium model as fallback...")
                try:
                    response = client.chat.complete(
                        model="mistral-medium",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_input}
                        ],
                        max_tokens=1000,
                        temperature=0.7,
                        timeout_ms=30000  # Add 30 second timeout
                    )
                    logger.info("mistral-medium response received successfully")
                except Exception as fallback_error:
                    logger.error(f"Both Mistral models failed. Mistral-medium error: {str(fallback_error)}")
                    logger.error(f"Mistral API key configured: {'Yes' if os.getenv('MISTRAL_API_KEY') else 'No'}")

                    # Check if it's a quota error
                    if 'insufficient_quota' in str(fallback_error) or '429' in str(fallback_error):
                        return Response({
                            'reply': '🤖 Our AI assistant is currently experiencing high demand and is temporarily unavailable. However, you can still search for flights directly!\n\n✈️ Try asking me about flights like:\n• "Find flights from Nairobi to London"\n• "Show me flights to Dubai"\n• "I need a flight from NBO to JNB"\n\nI can help you find and book flights even without the AI chat feature!',
                            'is_authenticated': is_authenticated,
                            'ai_unavailable': True
                        }, status=503)
                    else:
                        return Response({
                            'reply': 'Sorry, our AI assistant is temporarily unavailable due to technical issues. Please try again later.',
                            'is_authenticated': is_authenticated
                        }, status=503)

            reply = response.choices[0].message.content

            # Check if the response contains signup request
            signup_data = None
            if 'SIGNUP_REQUEST:' in reply:
                try:
                    # Extract signup data from the response
                    signup_start = reply.find('SIGNUP_REQUEST:') + len('SIGNUP_REQUEST:')
                    signup_line = reply[signup_start:].split('\n')[0].strip()
                    parts = signup_line.split()

                    if len(parts) >= 5:
                        signup_data = {
                            'first_name': parts[0],
                            'last_name': parts[1],
                            'username': parts[2],
                            'email': parts[3],
                            'password': ' '.join(parts[4:])  # Join remaining parts for password
                        }

                        # Attempt to create the user
                        logger.info(f"Attempting to create user: {signup_data}")
                        try:
                            if User.objects.filter(username=signup_data['username']).exists():
                                logger.warning(f"Username {signup_data['username']} already exists")
                                reply = "Sorry, that username is already taken. Please try a different username."
                            elif User.objects.filter(email=signup_data['email']).exists():
                                logger.warning(f"Email {signup_data['email']} already exists")
                                reply = "Sorry, that email is already registered. Please try a different email."
                            else:
                                with transaction.atomic():
                                    logger.info("Creating user...")
                                    user = User.objects.create_user(
                                        username=signup_data['username'],
                                        email=signup_data['email'],
                                        password=signup_data['password'],
                                        first_name=signup_data['first_name'],
                                        last_name=signup_data['last_name']
                                    )
                                    logger.info(f"User created with ID: {user.id}")

                                    # Create user profile
                                    logger.info("Creating user profile...")
                                    UserProfile.objects.create(user=user)
                                    logger.info("User profile created")

                                    # Create token
                                    token, created = Token.objects.get_or_create(user=user)
                                    logger.info(f"Token created: {token.key}")

                                # Send welcome email (outside transaction since email failure shouldn't rollback user creation)
                                try:
                                    send_mail(
                                        subject='Welcome to Bodrless!',
                                        message=f"Hi {signup_data['first_name'] or signup_data['username']},\n\nThank you for signing up to Bodrless. Your account has been created successfully!\n\nHappy travels!\nThe Bodrless Team",
                                        from_email=None,
                                        recipient_list=[signup_data['email']],
                                        fail_silently=True,
                                    )
                                    logger.info("Welcome email sent")
                                except Exception as e:
                                    logger.warning(f"Failed to send welcome email: {e}")

                                reply = f"Welcome to Bodrless officially! 🎉 You can now continue booking your amazing travel plans. Your account has been created successfully."

                        except Exception as e:
                            logger.error(f"User creation error: {e}")
                            logger.error(f"Error details: {str(e)}")
                            import traceback
                            logger.error(f"Traceback: {traceback.format_exc()}")
                            reply = "Sorry, there was an error creating your account. Please try again."

                    # Remove the signup request from the reply
                    signup_end = reply.find('\n', signup_start)
                    if signup_end == -1:
                        reply = reply[:signup_start-len('SIGNUP_REQUEST:')].strip()
                    else:
                        reply = reply[:signup_start-len('SIGNUP_REQUEST:')] + reply[signup_end:]

                except Exception as e:
                    logger.warning(f"Failed to parse signup request: {e}")
                    reply = "Sorry, there was an error processing your sign up request. Please try again."

            # Check if the response contains booking details
            booking_details = None
            if 'BOOKING_DETAILS:' in reply:
                try:
                    # Extract booking details from the response (exact order: full_name, id_number, passport_number, email)
                    booking_start = reply.find('BOOKING_DETAILS:') + len('BOOKING_DETAILS:')
                    booking_line = reply[booking_start:].split('\n')[0].strip()
                    parts = booking_line.split()

                    if len(parts) >= 4:
                        # Handle the case where full name might have spaces
                        # The format is: full_name id_number passport_number email
                        # But full_name can have spaces, so we need to be careful
                        # Clean up the parts by removing any brackets or quotes
                        cleaned_parts = [part.strip('[]"\'') for part in parts]

                        booking_details = {
                            'full_name': ' '.join(cleaned_parts[:-3]),  # Everything except last 3 parts
                            'id_number': cleaned_parts[-3],  # Third from last
                            'passport_number': cleaned_parts[-2],  # Second from last
                            'email': cleaned_parts[-1]  # Last part
                        }

                        # Save booking to database
                        try:
                            # For anonymous users, we need to handle the booking differently
                            # since FlightBooking requires a user. Let's create a temporary user or handle differently
                            # Create a sample flight search for this booking (you may need to adjust this)
                            flight_search = FlightSearch.objects.create(
                                user=request.user if request.user.is_authenticated else None,
                                from_location='NBO',  # Sample data - you may need to get this from context
                                to_location='LON',   # Sample data - you may need to get this from context
                                departure_date='2025-12-01',  # Sample data
                                adult_count=1
                            )

                            # Create the booking
                            booking = FlightBooking.objects.create(
                                user=request.user if request.user.is_authenticated else None,
                                flight_search=flight_search,
                                flight_data={'sample': 'flight_data'},  # You may need to get actual flight data
                                booking_reference=f'BL{booking_details["id_number"][-4:]}{FlightBooking.objects.count() + 1:04d}',
                                total_amount=500.00,  # Sample amount
                                currency='USD',
                                passenger_details=booking_details,
                                status='confirmed'
                            )

                            logger.info(f"Booking created: {booking.booking_reference}")
                            reply = f"Perfect! I've received your booking details and created your booking. 📋\n\nBooking Reference: {booking.booking_reference}\n👤 Name: {booking_details['full_name']}\n🆔 ID: {booking_details['id_number']}\n🛂 Passport: {booking_details['passport_number']}\n📧 Email: {booking_details['email']}\n\nA confirmation email has been sent to your email address.\n\nWhich payment method would you like to use?"

                            logger.info(f"Booking created: {booking.booking_reference}")

                            # Send confirmation email
                            try:
                                send_mail(
                                    subject='Booking Confirmation - Bodrless',
                                    message=f'''Dear {booking_details['full_name']},

Thank you for beginning your booking with Bodrless!

Booking Details:
- Booking Reference: {booking.booking_reference}
- Passenger: {booking_details['full_name']}
- ID Number: {booking_details['id_number']}
- Passport Number: {booking_details['passport_number']}
- Email: {booking_details['email']}
- Total Amount: ${booking.total_amount} {booking.currency}

Flight Information:
- From: Nairobi (NBO)
- To: London (LON)
- Date: December 1, 2025

Your booking has been confirmed. We will send you further details shortly.

Happy travels!
The Bodrless Team
''',
                                    from_email=None,
                                    recipient_list=[booking_details['email']],
                                    fail_silently=True,
                                )
                                logger.info(f"Confirmation email sent to {booking_details['email']}")
                            except Exception as e:
                                logger.warning(f"Failed to send confirmation email: {e}")

                            reply = f"Perfect! I've received your booking details and created your booking. 📋\n\nBooking Reference: {booking.booking_reference}\n👤 Name: {booking_details['full_name']}\n🆔 ID: {booking_details['id_number']}\n🛂 Passport: {booking_details['passport_number']}\n📧 Email: {booking_details['email']}\n\nA confirmation email has been sent to your email address.\n\nWhich payment method would you like to use?"

                        except Exception as e:
                            logger.error(f"Failed to create booking: {e}")
                            reply = "Sorry, there was an error creating your booking. Please try again."

                    # Remove the booking details from the reply
                    booking_end = reply.find('\n', booking_start)
                    if booking_end == -1:
                        reply = reply[:booking_start-len('BOOKING_DETAILS:')].strip()
                    else:
                        reply = reply[:booking_start-len('BOOKING_DETAILS:')] + reply[booking_end:]

                except Exception as e:
                    logger.warning(f"Failed to parse booking details: {e}")
                    reply = "Sorry, there was an error processing your booking details. Please try again."

            # Check if the response contains payment options
            payment_options = None
            if 'PAYMENT_OPTIONS:' in reply:
                try:
                    # Extract payment options from the response
                    payment_start = reply.find('PAYMENT_OPTIONS:') + len('PAYMENT_OPTIONS:')
                    payment_end = reply.find('END_PAYMENT_OPTIONS')
                    if payment_end == -1:
                        payment_end = len(reply)

                    payment_json = reply[payment_start:payment_end].strip()
                    payment_options = json.loads(payment_json)

                    # Remove the payment options from the reply
                    reply = reply.replace(reply[payment_start-len('PAYMENT_OPTIONS:'):payment_end], '').strip()

                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Failed to parse payment options JSON: {e}")
                    payment_options = None

            # Check if the response contains structured travel options data
            travel_options = None
            if 'TRAVEL_OPTIONS:' in reply:
                try:
                    # Extract JSON data from the response
                    options_start = reply.find('TRAVEL_OPTIONS:') + len('TRAVEL_OPTIONS:')
                    options_end = reply.find('END_TRAVEL_OPTIONS')
                    if options_end == -1:
                        options_end = len(reply)

                    options_json = reply[options_start:options_end].strip()
                    travel_options = json.loads(options_json)

                    # Remove the structured data from the reply text
                    reply = reply.replace(reply[options_start-len('TRAVEL_OPTIONS:'):options_end], '').strip()
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Failed to parse travel options JSON: {e}")
                    travel_options = None

            logger.info(f"✅ Chat response sent successfully. Reply length: {len(reply)} chars")
            return Response({
                'reply': reply,
                'is_authenticated': is_authenticated,
                'travel_options': travel_options,
                'payment_options': payment_options
            })

        except Exception as e:
            logger.error(f"❌ Chat API error: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return Response({'reply': 'Sorry, I encountered an unexpected error. Please try again later.'}, status=500)


@method_decorator(csrf_exempt, name='dispatch')
class FlightSearchView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        try:
            data = json.loads(request.body)

            # Extract flight search parameters
            from_location = data.get('from_location')
            to_location = data.get('to_location')
            date = data.get('date')
            adult_count = data.get('adult_count', 1)
            child_count = data.get('child_count', 0)
            infant_count = data.get('infant_count', 0)
            cabin_class = data.get('cabin_class', 'economy')

            # Save search to database (only if user is authenticated)
            flight_search = None
            if request.user.is_authenticated:
                flight_search = FlightSearch.objects.create(
                    user=request.user,
                    from_location=from_location,
                    to_location=to_location,
                    departure_date=date,
                    adult_count=adult_count,
                    child_count=child_count,
                    infant_count=infant_count,
                    cabin_class=cabin_class
                )

            # Perform flight search
            results = search_flights(
                from_location=from_location,
                to_location=to_location,
                date=date,
                adult_count=adult_count,
                child_count=child_count,
                infant_count=infant_count,
                cabin_class=cabin_class
            )

            # Save results to database (only if flight_search was created)
            if flight_search:
                flight_search.search_results = results
                flight_search.save()

            return Response({
                'success': True,
                'flights': results.get('offers', []),
                'message': 'Flights retrieved successfully'
            })

        except Exception as e:
            logger.error(f"Flight search error: {e}")
            return Response({
                'success': False,
                'flights': [],
                'message': f'Error searching flights: {str(e)}'
            }, status=500)


@method_decorator(csrf_exempt, name='dispatch')
class HealthView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        return Response({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'service': 'Bodrless Django API'
        })


@method_decorator(csrf_exempt, name='dispatch')
class BookingView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        try:
            data = json.loads(request.body)
            flight_data = data.get('flight_data')
            passenger_details = data.get('passenger_details', {})

            # Debug logging to see the actual structure
            logger.info(f"DEBUG: Received booking request data: {json.dumps(data, indent=2)}")
            logger.info(f"DEBUG: Flight data type: {type(flight_data)}")
            logger.info(f"DEBUG: Flight data content: {flight_data}")

            if not flight_data:
                return Response({'error': 'Flight data is required'}, status=400)

            # Extract origin and destination from various possible sources
            origin = None
            destination = None

            # Try direct field names first
            origin = flight_data.get('origin') or flight_data.get('from_location') or flight_data.get('departure')
            destination = flight_data.get('destination') or flight_data.get('to_location') or flight_data.get('arrival')

            # If not found directly, try to extract from title or description
            if not origin or not destination:
                title = flight_data.get('title', '')
                description = flight_data.get('description', '')

                # Extract from title like "Economy Flight NBO → SOU"
                if '→' in title and not origin or not destination:
                    parts = title.split('→')
                    if len(parts) == 2:
                        potential_origin = parts[0].strip().split()[-1]  # Get last part (airport code)
                        potential_destination = parts[1].strip().split()[-1]  # Get last part (airport code)

                        if not origin:
                            origin = potential_origin
                        if not destination:
                            destination = potential_destination

                # Extract from description like "flight from Nairobi (NBO) to South Africa (SOU)"
                if not origin or not destination:
                    desc_text = f"{description} {flight_data.get('detailed_info', {}).get('route', '')}"
                    import re

                    # Look for pattern: "from City (CODE)" and "to City (CODE)"
                    from_match = re.search(r'from\s+[^)]+\s*\(([^)]+)\)', desc_text)
                    to_match = re.search(r'to\s+[^)]+\s*\(([^)]+)\)', desc_text)

                    if from_match and not origin:
                        origin = from_match.group(1)
                    if to_match and not destination:
                        destination = to_match.group(1)

            # If still not found, try to extract from route in detailed_info
            if not origin or not destination:
                route = flight_data.get('detailed_info', {}).get('route', '')
                if '→' in route and not origin or not destination:
                    parts = route.split('→')
                    if len(parts) == 2:
                        potential_origin = parts[0].strip()
                        potential_destination = parts[1].strip()

                        if not origin:
                            origin = potential_origin
                        if not destination:
                            destination = potential_destination

            if not origin:
                return Response({
                    'error': 'Missing required field: origin',
                    'details': 'Could not extract origin from flight data. Expected fields: origin, from_location, departure, or extractable from title/description'
                }, status=400)

            if not destination:
                return Response({
                    'error': 'Missing required field: destination',
                    'details': 'Could not extract destination from flight data. Expected fields: destination, to_location, arrival, or extractable from title/description'
                }, status=400)

            # Update flight_data with standardized field names for consistency
            flight_data = flight_data.copy()
            flight_data['origin'] = origin
            flight_data['destination'] = destination

            # Parse and validate price using utility function
            try:
                total_amount, detected_currency = parse_and_validate_price(
                    flight_data.get('price', '0'),
                    'flight price'
                )
            except ValueError as e:
                logger.error(f"Price validation error: {e}")
                return Response({
                    'error': 'Invalid price format',
                    'details': str(e)
                }, status=400)

            # Create flight search record
            try:
                flight_search = FlightSearch.objects.create(
                    user=request.user if request.user.is_authenticated else None,
                    from_location=flight_data.get('origin', 'NBO')[:3],
                    to_location=flight_data.get('destination', 'LHR')[:3],
                    departure_date=datetime.now().date(),
                    adult_count=1,
                    cabin_class=flight_data.get('cabin_class', 'economy')
                )
            except Exception as e:
                logger.error(f"Flight search creation error: {e}")
                return Response({
                    'error': 'Failed to create flight search record',
                    'details': 'Database error while creating flight search'
                }, status=500)

            # Generate booking reference
            import random
            booking_ref = f"BL{random.randint(100000, 999999)}"

            # Create booking
            try:
                booking = FlightBooking.objects.create(
                    user=request.user if request.user.is_authenticated else None,
                    flight_search=flight_search,
                    flight_data=flight_data,
                    booking_reference=booking_ref,
                    total_amount=total_amount,
                    currency=detected_currency,
                    passenger_details=passenger_details,
                    status='pending'
                )
            except Exception as e:
                logger.error(f"Booking creation error: {e}")
                # Clean up flight search if booking creation fails
                try:
                    flight_search.delete()
                except:
                    pass
                return Response({
                    'error': 'Failed to create booking',
                    'details': 'Database error while creating booking record'
                }, status=500)

            logger.info(f"✅ Booking created successfully: {booking_ref} for user {request.user.username if request.user.is_authenticated else 'anonymous'}")

            return Response({
                'success': True,
                'booking_reference': booking_ref,
                'booking_id': booking.id,
                'total_amount': float(booking.total_amount),
                'currency': booking.currency,
                'message': 'Booking created successfully. Please select a payment method.'
            })

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in booking: {e}")
            return Response({
                'error': 'Invalid JSON format',
                'details': 'Request body must contain valid JSON data'
            }, status=400)

        except Exception as e:
            logger.error(f"Unexpected booking creation error: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return Response({
                'error': 'Failed to create booking',
                'details': 'An unexpected error occurred during booking creation'
            }, status=500)


@method_decorator(csrf_exempt, name='dispatch')
class PaymentOptionsView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        try:
            # Get IntaSend configuration
            intasend_config = {
                'name': 'IntaSend',
                'type': 'intasend',
                'description': 'Pay with IntaSend (M-Pesa, Card, Bank)',
                'currencies': ['KES', 'USD'],
                'features': ['M-Pesa', 'Card payments', 'Bank transfers'],
                'setup_required': False
            }

            # M-Pesa configuration
            mpesa_config = {
                'name': 'M-Pesa',
                'type': 'mpesa',
                'description': 'Pay with M-Pesa mobile money',
                'currencies': ['KES'],
                'features': ['Mobile money', 'Instant payment'],
                'setup_required': False
            }

            # Stripe configuration
            stripe_config = {
                'name': 'Stripe',
                'type': 'stripe',
                'description': 'Pay with credit/debit card via Stripe',
                'currencies': ['USD', 'EUR', 'GBP'],
                'features': ['Credit cards', 'Debit cards', 'Digital wallets'],
                'setup_required': True
            }

            return Response({
                'success': True,
                'payment_options': [
                    intasend_config,
                    mpesa_config,
                    stripe_config
                ],
                'recommended': 'intasend'
            })

        except Exception as e:
            logger.error(f"Payment options error: {e}")
            return Response({'error': 'Failed to load payment options'}, status=500)


@method_decorator(csrf_exempt, name='dispatch')
class TrainBookingView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        try:
            data = json.loads(request.body)
            train_data = data.get('train_data')
            user_input = data.get('user_input', '')

            if not train_data:
                return Response({'error': 'Train data is required'}, status=400)

            # Extract origin and destination from train data
            origin = train_data.get('origin', 'NBO')
            destination = train_data.get('destination', 'MBA')

            # Parse and validate price using utility function
            try:
                total_amount, detected_currency = parse_and_validate_price(
                    train_data.get('price', '0').replace(',', ''),
                    'train fare'
                )
            except ValueError as e:
                logger.error(f"Price validation error: {e}")
                return Response({
                    'error': 'Invalid price format',
                    'details': str(e)
                }, status=400)

            # Extract cities from user input for context
            from_city = 'Nairobi'
            to_city = 'Mombasa'

            if 'from' in user_input.lower() and 'to' in user_input.lower():
                import re
                pattern = r'from\s+(.+?)\s+to\s+(.+)'
                match = re.search(pattern, user_input.lower())
                if match:
                    from_city = match.group(1).strip().title()
                    to_city = match.group(2).strip().title()

            # Create train search record
            try:
                train_search = FlightSearch.objects.create(
                    user=request.user if request.user.is_authenticated else None,
                    from_location=origin,
                    to_location=destination,
                    departure_date=datetime.now().date(),
                    adult_count=1,
                    cabin_class=train_data.get('class', 'economy')
                )
            except Exception as e:
                logger.error(f"Train search creation error: {e}")
                return Response({
                    'error': 'Failed to create train search record',
                    'details': 'Database error while creating train search'
                }, status=500)

            # Generate booking reference
            import random
            booking_ref = f"TR{random.randint(100000, 999999)}"

            # Create train booking
            try:
                booking = FlightBooking.objects.create(
                    user=request.user if request.user.is_authenticated else None,
                    flight_search=train_search,
                    flight_data=train_data,
                    booking_reference=booking_ref,
                    total_amount=total_amount,
                    currency=detected_currency,
                    passenger_details={},
                    status='pending'
                )
            except Exception as e:
                logger.error(f"Train booking creation error: {e}")
                # Clean up train search if booking creation fails
                try:
                    train_search.delete()
                except:
                    pass
                return Response({
                    'error': 'Failed to create train booking',
                    'details': 'Database error while creating train booking record'
                }, status=500)

            # Get payment options
            payment_options_response = {
                'success': True,
                'payment_options': [
                    {
                        'name': 'IntaSend',
                        'type': 'intasend',
                        'description': 'Pay with IntaSend (M-Pesa, Card, Bank)',
                        'currencies': ['KES', 'USD'],
                        'features': ['M-Pesa', 'Card payments', 'Bank transfers'],
                        'setup_required': False
                    },
                    {
                        'name': 'M-Pesa',
                        'type': 'mpesa',
                        'description': 'Pay with M-Pesa mobile money',
                        'currencies': ['KES'],
                        'features': ['Mobile money', 'Instant payment'],
                        'setup_required': False
                    },
                    {
                        'name': 'Stripe',
                        'type': 'stripe',
                        'description': 'Pay with credit/debit card via Stripe',
                        'currencies': ['USD', 'EUR', 'GBP'],
                        'features': ['Credit cards', 'Debit cards', 'Digital wallets'],
                        'setup_required': True
                    }
                ],
                'recommended': 'intasend',
                'booking_reference': booking_ref,
                'booking_id': booking.id,
                'amount': booking.total_amount,
                'currency': booking.currency
            }

            logger.info(f"✅ Train booking initiated successfully: {booking_ref} for user {request.user.username if request.user.is_authenticated else 'anonymous'}")

            return Response({
                'success': True,
                'message': f'Train selected! Please complete your booking for {from_city} to {to_city}.',
                'booking_reference': booking_ref,
                'booking_id': booking.id,
                'train_details': train_data,
                'payment_options': payment_options_response['payment_options'],
                'next_step': 'passenger_details'
            })

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in train booking: {e}")
            return Response({
                'error': 'Invalid JSON format',
                'details': 'Request body must contain valid JSON data'
            }, status=400)

        except Exception as e:
            logger.error(f"Unexpected train booking error: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return Response({
                'error': 'Failed to initiate train booking',
                'details': 'An unexpected error occurred during train booking initiation'
            }, status=500)


@method_decorator(csrf_exempt, name='dispatch')
class FlightBookingView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        try:
            data = json.loads(request.body)
            flight_data = data.get('flight_data')
            user_input = data.get('user_input', '')

            if not flight_data:
                return Response({'error': 'Flight data is required'}, status=400)

            # Extract origin and destination from various possible sources
            origin = None
            destination = None

            # Try direct field names first
            origin = flight_data.get('origin') or flight_data.get('from_location') or flight_data.get('departure')
            destination = flight_data.get('destination') or flight_data.get('to_location') or flight_data.get('arrival')

            # If not found directly, try to extract from title or description
            if not origin or not destination:
                title = flight_data.get('title', '')
                description = flight_data.get('description', '')

                # Extract from title like "Economy Flight NBO → SOU"
                if '→' in title and not origin or not destination:
                    parts = title.split('→')
                    if len(parts) == 2:
                        potential_origin = parts[0].strip().split()[-1]  # Get last part (airport code)
                        potential_destination = parts[1].strip().split()[-1]  # Get last part (airport code)

                        if not origin:
                            origin = potential_origin
                        if not destination:
                            destination = potential_destination

                # Extract from description like "flight from Nairobi (NBO) to South Africa (SOU)"
                if not origin or not destination:
                    desc_text = f"{description} {flight_data.get('detailed_info', {}).get('route', '')}"
                    import re

                    # Look for pattern: "from City (CODE)" and "to City (CODE)"
                    from_match = re.search(r'from\s+[^)]+\s*\(([^)]+)\)', desc_text)
                    to_match = re.search(r'to\s+[^)]+\s*\(([^)]+)\)', desc_text)

                    if from_match and not origin:
                        origin = from_match.group(1)
                    if to_match and not destination:
                        destination = to_match.group(1)

            # If still not found, try to extract from route in detailed_info
            if not origin or not destination:
                route = flight_data.get('detailed_info', {}).get('route', '')
                if '→' in route and not origin or not destination:
                    parts = route.split('→')
                    if len(parts) == 2:
                        potential_origin = parts[0].strip()
                        potential_destination = parts[1].strip()

                        if not origin:
                            origin = potential_origin
                        if not destination:
                            destination = potential_destination

            if not origin:
                return Response({
                    'error': 'Missing required field: origin',
                    'details': 'Could not extract origin from flight data. Expected fields: origin, from_location, departure, or extractable from title/description'
                }, status=400)

            if not destination:
                return Response({
                    'error': 'Missing required field: destination',
                    'details': 'Could not extract destination from flight data. Expected fields: destination, to_location, arrival, or extractable from title/description'
                }, status=400)

            # Update flight_data with standardized field names for consistency
            flight_data = flight_data.copy()
            flight_data['origin'] = origin
            flight_data['destination'] = destination

            # Parse and validate price using utility function
            try:
                total_amount, detected_currency = parse_and_validate_price(
                    flight_data.get('price', '0'),
                    'flight price'
                )
            except ValueError as e:
                logger.error(f"Price validation error: {e}")
                return Response({
                    'error': 'Invalid price format',
                    'details': str(e)
                }, status=400)

            # Extract cities from user input for context
            from_city = 'Nairobi'
            to_city = 'Mombasa'

            if 'from' in user_input.lower() and 'to' in user_input.lower():
                import re
                pattern = r'from\s+(.+?)\s+to\s+(.+)'
                match = re.search(pattern, user_input.lower())
                if match:
                    from_city = match.group(1).strip().title()
                    to_city = match.group(2).strip().title()

            # Create flight search record
            try:
                flight_search = FlightSearch.objects.create(
                    user=request.user if request.user.is_authenticated else None,
                    from_location=flight_data.get('origin', 'NBO')[:3],
                    to_location=flight_data.get('destination', 'MBA')[:3],
                    departure_date=datetime.now().date(),
                    adult_count=1,
                    cabin_class=flight_data.get('cabin_class', 'economy')
                )
            except Exception as e:
                logger.error(f"Flight search creation error: {e}")
                return Response({
                    'error': 'Failed to create flight search record',
                    'details': 'Database error while creating flight search'
                }, status=500)

            # Generate booking reference
            import random
            booking_ref = f"BL{random.randint(100000, 999999)}"

            # Create booking
            try:
                booking = FlightBooking.objects.create(
                    user=request.user if request.user.is_authenticated else None,
                    flight_search=flight_search,
                    flight_data=flight_data,
                    booking_reference=booking_ref,
                    total_amount=total_amount,
                    currency=detected_currency,
                    passenger_details={},
                    status='pending'
                )
            except Exception as e:
                logger.error(f"Booking creation error: {e}")
                # Clean up flight search if booking creation fails
                try:
                    flight_search.delete()
                except:
                    pass
                return Response({
                    'error': 'Failed to create booking',
                    'details': 'Database error while creating booking record'
                }, status=500)

            # Get payment options
            payment_options_response = {
                'success': True,
                'payment_options': [
                    {
                        'name': 'IntaSend',
                        'type': 'intasend',
                        'description': 'Pay with IntaSend (M-Pesa, Card, Bank)',
                        'currencies': ['KES', 'USD'],
                        'features': ['M-Pesa', 'Card payments', 'Bank transfers'],
                        'setup_required': False
                    },
                    {
                        'name': 'M-Pesa',
                        'type': 'mpesa',
                        'description': 'Pay with M-Pesa mobile money',
                        'currencies': ['KES'],
                        'features': ['Mobile money', 'Instant payment'],
                        'setup_required': False
                    },
                    {
                        'name': 'Stripe',
                        'type': 'stripe',
                        'description': 'Pay with credit/debit card via Stripe',
                        'currencies': ['USD', 'EUR', 'GBP'],
                        'features': ['Credit cards', 'Debit cards', 'Digital wallets'],
                        'setup_required': True
                    }
                ],
                'recommended': 'intasend',
                'booking_reference': booking_ref,
                'booking_id': booking.id,
                'amount': booking.total_amount,
                'currency': booking.currency
            }

            logger.info(f"✅ Flight booking initiated successfully: {booking_ref} for user {request.user.username if request.user.is_authenticated else 'anonymous'}")

            return Response({
                'success': True,
                'message': f'Flight selected! Please complete your booking for {from_city} to {to_city}.',
                'booking_reference': booking_ref,
                'booking_id': booking.id,
                'flight_details': flight_data,
                'payment_options': payment_options_response['payment_options'],
                'next_step': 'passenger_details'
            })

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in flight booking: {e}")
            return Response({
                'error': 'Invalid JSON format',
                'details': 'Request body must contain valid JSON data'
            }, status=400)

        except Exception as e:
            logger.error(f"Unexpected flight booking error: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return Response({
                'error': 'Failed to initiate booking',
                'details': 'An unexpected error occurred during booking initiation'
            }, status=500)


@method_decorator(csrf_exempt, name='dispatch')
class IntaSendPaymentView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        try:
            data = json.loads(request.body)
            booking_id = data.get('booking_id')
            amount = data.get('amount')
            currency = data.get('currency', 'KES')
            phone_number = data.get('phone_number', '')
            email = data.get('email', '')

            logger.info("🔍 IntaSend Payment Request Debug:")
            logger.info(f"  - Booking ID: {booking_id}")
            logger.info(f"  - Amount: {amount}")
            logger.info(f"  - Currency: {currency}")
            logger.info(f"  - Phone: {phone_number}")
            logger.info(f"  - Email: {email}")

            if not booking_id or not amount:
                logger.error("❌ Missing required parameters: booking_id or amount")
                return Response({'error': 'Booking ID and amount are required'}, status=400)

            # Get booking details
            try:
                booking = FlightBooking.objects.get(id=booking_id)
                logger.info(f"✅ Found booking: {booking.booking_reference} - {booking.status}")

                # If phone number not provided in request, try to get it from database
                if not phone_number:
                    # First try to get from user profile
                    if booking.user:
                        try:
                            user_profile = UserProfile.objects.get(user=booking.user)
                            if user_profile.phone_number:
                                phone_number = user_profile.phone_number
                                logger.info(f"📱 Retrieved phone number from user profile: {phone_number}")
                        except UserProfile.DoesNotExist:
                            logger.warning("⚠️ User profile not found")

                    # If not found in profile, try to get from passenger details
                    if not phone_number and booking.passenger_details:
                        passenger_phone = booking.passenger_details.get('phone_number')
                        if passenger_phone:
                            phone_number = passenger_phone
                            logger.info(f"📱 Retrieved phone number from passenger details: {phone_number}")
                        else:
                            logger.warning("⚠️ No phone number found in passenger details")

                # If email not provided, try to get it from passenger details or user
                if not email:
                    if booking.passenger_details and booking.passenger_details.get('email'):
                        email = booking.passenger_details.get('email')
                        logger.info(f"📧 Retrieved email from passenger details: {email}")
                    elif booking.user and booking.user.email:
                        email = booking.user.email
                        logger.info(f"📧 Retrieved email from user account: {email}")

            except FlightBooking.DoesNotExist:
                logger.error(f"❌ Booking not found: {booking_id}")
                return Response({'error': 'Booking not found'}, status=404)

            # Check if IntaSend is enabled in settings
            if not settings.INTASEND_ENABLED:
                logger.warning("⚠️ IntaSend payment gateway is disabled in settings")
                return Response({
                    'error': 'Payment gateway temporarily unavailable',
                    'details': 'IntaSend payment integration is currently disabled. Please contact support for assistance.',
                    'suggestion': 'To enable payments, set valid INTASEND_API_KEY and INTASEND_PUBLISHABLE_KEY in your environment variables.',
                    'setup_instructions': 'Get your API keys from https://sandbox.intasend.com and add them to your .env file'
                }, status=503)

            # IntaSend SDK integration
            if not INTASEND_AVAILABLE or APIService is None:
                logger.error("❌ IntaSend SDK not available")
                logger.error(f"  - INTASEND_AVAILABLE: {INTASEND_AVAILABLE}")
                logger.error(f"  - APIService: {APIService}")
                return Response({
                    'error': 'Payment gateway not available',
                    'details': 'IntaSend SDK not properly configured'
                }, status=500)

            try:
                intasend_api_key = settings.INTASEND_API_KEY or "ISSecretKey_test_1234567890"
                intasend_publishable_key = settings.INTASEND_PUBLISHABLE_KEY or "ISPubKey_test_12345678-1234-1234-1234-123456789012"

                logger.info("🔑 IntaSend Configuration Debug:")
                logger.info(f"  - API Key (first 20 chars): {intasend_api_key[:20]}...")
                logger.info(f"  - API Key type: {'Secret' if 'ISSecretKey' in intasend_api_key else 'Publishable'}")
                logger.info(f"  - Publishable Key (first 20 chars): {intasend_publishable_key[:20]}...")
                logger.info(f"  - Is test key: {intasend_api_key.startswith('ISSecretKey_test_')}")

                # Validate key format
                if not intasend_api_key.startswith('ISSecretKey'):
                    logger.warning("⚠️ API key appears to be a publishable key instead of secret key")
                    logger.warning("  - Expected format: ISSecretKey_test_... or ISSecretKey_live_...")
                    logger.warning("  - Current format: Publishable key format")

                if not intasend_publishable_key.startswith('ISPubKey'):
                    logger.warning("⚠️ Publishable key format incorrect")
                    logger.warning("  - Expected format: ISPubKey_test_... or ISPubKey_live_...")

                # Use test mode for sandbox environment, live mode for production
                # Since this is a live key, set test=False
                is_test_key = intasend_api_key.startswith('ISSecretKey_test_')
                logger.info(f"Initializing IntaSend service with test mode: {is_test_key}")
                logger.info(f"API key prefix: {intasend_api_key[:20]}...")

                # Try to initialize with additional parameters
                try:
                    intasend_publishable_key = settings.INTASEND_PUBLISHABLE_KEY or "ISPubKey_test_12345678-1234-1234-1234-123456789012"
                    logger.info(f"🔑 Using publishable key: {intasend_publishable_key[:20]}...")

                    logger.info("🚀 Attempting IntaSend service initialization with publishable key...")
                    intasend_service = APIService(
                        token=intasend_api_key,
                        publishable_key=intasend_publishable_key,
                        test=is_test_key
                    )
                    logger.info("✅ IntaSend service initialized with publishable key")
                except Exception as init_error:
                    logger.error(f"❌ Failed to initialize IntaSend service with publishable key: {init_error}")
                    logger.error(f"  - Error type: {type(init_error).__name__}")
                    logger.info("🔄 Falling back to basic initialization...")
                    # Fallback to basic initialization
                    intasend_service = APIService(token=intasend_api_key, test=is_test_key)
                    logger.info("✅ IntaSend service initialized with fallback method")

                logger.info("🎯 IntaSend service initialized successfully - ready for payment processing")

                # Check if phone number is provided for M-Pesa STK Push
                if phone_number and (phone_number.startswith('+254') or phone_number.startswith('254')):
                    # Use M-Pesa STK Push for Kenyan phone numbers
                    logger.info(f"📱 Using M-Pesa STK Push for phone: {phone_number}")
                    logger.info(f"  - Cleaned phone number: {phone_number.replace('+', '')}")

                    try:
                        logger.info("💳 Initiating M-Pesa STK Push...")
                        response = intasend_service.collect.mpesa_stk_push(
                            phone_number=int(phone_number.replace('+', '')),  # Remove + prefix
                            email=email,
                            amount=float(amount),
                            currency=currency,
                            narrative=f"Flight booking - {booking.booking_reference}"
                        )
                        logger.info(f"✅ M-Pesa STK Push initiated successfully")
                        logger.info(f"  - Response ID: {response.get('id', 'N/A')}")
                        logger.info(f"  - Checkout URL: {response.get('checkout_url', 'N/A')}")

                        # Update booking status
                        booking.status = 'pending'
                        booking.save()
                        logger.info(f"💾 Updated booking {booking.booking_reference} status to pending")

                        # Send payment initiation email
                        try:
                            # Get passenger details and flight data from booking
                            passenger_details = booking.passenger_details or {}
                            flight_data = booking.flight_data or {}

                            # Extract flight information
                            flight_title = flight_data.get('title', flight_data.get('flight_number', 'Flight'))
                            flight_segments = flight_data.get('flight_segments', [{}])
                            first_segment = flight_segments[0] if flight_segments else {}

                            departure_airport = first_segment.get('departure_airport', flight_data.get('origin', 'NBO'))
                            arrival_airport = first_segment.get('arrival_airport', flight_data.get('destination', 'LHR'))
                            departure_time = first_segment.get('departure_time', flight_data.get('departure_time', 'TBD'))
                            arrival_time = first_segment.get('arrival_time', flight_data.get('arrival_time', 'TBD'))
                            airline = first_segment.get('operating_airline', flight_data.get('airline', 'Airline'))
                            aircraft = first_segment.get('aircraft_type', 'Aircraft')
                            duration = flight_data.get('duration', 'Duration')

                            initiation_html = f"""
                            <!DOCTYPE html>
                            <html>
                            <head>
                                <meta charset="utf-8">
                                <title>Payment Initiated - Bodrless</title>
                                <style>
                                    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); }}
                                    .container {{ max-width: 700px; margin: 0 auto; background: white; border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.15); overflow: hidden; }}
                                    .header {{ background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 50%, #1e40af 100%); color: white; padding: 25px; text-align: center; position: relative; }}
                                    .header::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="dots" width="20" height="20" patternUnits="userSpaceOnUse"><circle cx="10" cy="10" r="1" fill="rgba(255,255,255,0.1)"/></pattern></defs><rect width="100" height="100" fill="url(%23dots)"/></svg>'); pointer-events: none; }}
                                    .header h1 {{ margin: 0; font-size: 26px; font-weight: 700; text-shadow: 0 2px 4px rgba(0,0,0,0.2); }}
                                    .header p {{ margin: 8px 0 0 0; font-size: 16px; opacity: 0.9; }}
                                    .content {{ padding: 30px; }}
                                    .flight-card {{ background: linear-gradient(135deg, #f8fafc, #f1f5f9); border-radius: 12px; padding: 20px; margin: 20px 0; border-left: 4px solid #3b82f6; }}
                                    .flight-header {{ background: #3b82f6; color: white; padding: 15px; border-radius: 10px; margin-bottom: 20px; text-align: center; }}
                                    .flight-info {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 15px 0; }}
                                    .info-section {{ margin-bottom: 15px; }}
                                    .info-label {{ font-weight: 600; color: #475569; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }}
                                    .info-value {{ color: #1e293b; font-size: 16px; font-weight: 500; }}
                                    .route {{ font-size: 18px; font-weight: 700; color: #3b82f6; text-align: center; margin: 15px 0; }}
                                    .payment-info {{ background: linear-gradient(135deg, #eff6ff, #dbeafe); padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 4px solid #f59e0b; }}
                                    .status {{ text-align: center; margin: 20px 0; }}
                                    .status-badge {{ background: #f59e0b; color: white; padding: 8px 16px; border-radius: 20px; font-weight: 600; text-transform: uppercase; font-size: 14px; }}
                                    .footer {{ text-align: center; color: #64748b; font-size: 13px; margin-top: 30px; padding: 20px; background: #f8fafc; border-radius: 0 0 15px 15px; }}
                                    .highlight {{ background: linear-gradient(135deg, #fef3c7, #fde68a); padding: 15px; border-radius: 10px; border-left: 4px solid #f59e0b; margin: 20px 0; }}
                                </style>
                            </head>
                            <body>
                                <div class="container">
                                    <div class="header">
                                        <h1>💳 Payment Initiated</h1>
                                        <p>Your booking is being processed</p>
                                    </div>

                                    <div class="content">
                                        <div class="flight-card">
                                            <div class="flight-header">
                                                <h3 style="margin: 0; color: white;">✈️ Flight Information</h3>
                                            </div>

                                            <div style="text-align: center; margin-bottom: 20px;">
                                                <div class="route">
                                                    {departure_airport.split(' (')[0]} → {arrival_airport.split(' (')[0]}
                                                </div>
                                                <p style="color: #64748b; margin: 5px 0 0 0; font-size: 14px;">{flight_title} • {duration}</p>
                                            </div>

                                            <div class="flight-info">
                                                <div>
                                                    <div class="info-section">
                                                        <div class="info-label">Departure</div>
                                                        <div class="info-value">{departure_airport}</div>
                                                        <div style="color: #64748b; font-size: 14px;">{departure_time}</div>
                                                    </div>
                                                    <div class="info-section">
                                                        <div class="info-label">Airline</div>
                                                        <div class="info-value">{airline}</div>
                                                        <div style="color: #64748b; font-size: 14px;">{aircraft}</div>
                                                    </div>
                                                </div>
                                                <div>
                                                    <div class="info-section">
                                                        <div class="info-label">Arrival</div>
                                                        <div class="info-value">{arrival_airport}</div>
                                                        <div style="color: #64748b; font-size: 14px;">{arrival_time}</div>
                                                    </div>
                                                    <div class="info-section">
                                                        <div class="info-label">Duration</div>
                                                        <div class="info-value">{duration}</div>
                                                        <div style="color: #64748b; font-size: 14px;">Direct Flight</div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>

                                        <div class="payment-info">
                                            <h3 style="margin: 0 0 15px 0; color: #1e293b;">💳 Payment Details</h3>
                                            <div class="flight-info">
                                                <div>
                                                    <div class="info-section">
                                                        <div class="info-label">Booking Reference</div>
                                                        <div class="info-value" style="font-family: monospace; color: #3b82f6;">{booking.booking_reference}</div>
                                                    </div>
                                                    <div class="info-section">
                                                        <div class="info-label">Amount</div>
                                                        <div class="info-value" style="font-size: 18px; color: #059669; font-weight: 600;">{currency} {amount:,}</div>
                                                    </div>
                                                </div>
                                                <div>
                                                    <div class="info-section">
                                                        <div class="info-label">Payment Method</div>
                                                        <div class="info-value">M-Pesa STK Push</div>
                                                    </div>
                                                    <div class="info-section">
                                                        <div class="info-label">Status</div>
                                                        <div class="status">
                                                            <span class="status-badge">Pending</span>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>

                                        <div class="highlight">
                                            <h4 style="margin: 0 0 10px 0; color: #92400e;">📱 Next Steps</h4>
                                            <p style="margin: 5px 0; color: #78350f;">• Please check your phone for the M-Pesa payment prompt</p>
                                            <p style="margin: 5px 0; color: #78350f;">• Enter your M-Pesa PIN to complete the payment</p>
                                            <p style="margin: 5px 0; color: #78350f;">• You will receive your e-ticket once payment is confirmed</p>
                                        </div>
                                    </div>

                                    <div class="footer">
                                        <p>Thank you for choosing Bodrless for your journey!</p>
                                        <p>For support: support@bodrless.com | +254 700 000 000</p>
                                        <p style="margin-top: 15px; opacity: 0.8; font-size: 12px;">This is a payment confirmation. Your e-ticket will be sent separately upon payment completion.</p>
                                    </div>
                                </div>
                            </body>
                            </html>
                            """

                            send_mail(
                                subject=f'💳 Payment Initiated - Flight {flight_title} ({booking.booking_reference})',
                                message=f'''
Dear {passenger_details.get('full_name', passenger_details.get('first_name', 'Valued Customer'))},

Your payment has been initiated for your flight booking!

FLIGHT DETAILS:
- Flight: {flight_title}
- Route: {departure_airport.split(' (')[0]} → {arrival_airport.split(' (')[0]}
- Departure: {departure_time}
- Arrival: {arrival_time}
- Airline: {airline}
- Duration: {duration}

BOOKING DETAILS:
- Booking Reference: {booking.booking_reference}
- Amount: {currency} {amount:,}
- Payment Method: M-Pesa STK Push
- Status: Pending

Please check your phone and enter your M-Pesa PIN to complete the payment.

You will receive your beautiful e-ticket email once the payment is successfully processed.

Thank you for choosing Bodrless!
The Bodrless Team
''',
                                from_email=None,
                                recipient_list=[email],
                                fail_silently=True,
                                html_message=initiation_html
                            )
                            logger.info(f"Payment initiation email sent to {email}")
                        except Exception as e:
                            logger.warning(f"Failed to send payment initiation email: {e}")

                        return Response({
                            'success': True,
                            'payment_method': 'mpesa_stk',
                            'payment_id': response.get('id', ''),
                            'message': 'M-Pesa STK Push sent to your phone. Please check your phone and enter your PIN.',
                            'checkout_url': response.get('checkout_url', ''),
                            'status': 'pending'
                        })
                    except Exception as stk_error:
                        logger.error(f"❌ M-Pesa STK Push failed: {stk_error}")
                        logger.error(f"  - Error type: {type(stk_error).__name__}")

                        # Return proper error response for STK push failure
                        if 'authentication_failed' in str(stk_error) or 'Session expired' in str(stk_error):
                            return Response({
                                'error': 'Payment gateway authentication failed',
                                'details': 'Invalid or expired IntaSend API key. Please check your configuration.',
                                'suggestion': 'Use a valid IntaSend API key from https://sandbox.intasend.com',
                                'fallback_available': True
                            }, status=503)
                        else:
                            # Fall through to regular checkout for other errors
                            logger.info("🔄 Falling back to regular checkout due to STK Push failure")

                # Use regular checkout for other payment methods or fallback
                logger.info(f"💰 Creating regular payment checkout...")
                logger.info(f"  - Email: {email}")
                logger.info(f"  - Amount: {amount}")
                logger.info(f"  - Currency: {currency}")

                try:
                    response = intasend_service.collect.checkout(
                        email=email,
                        amount=float(amount),
                        currency=currency
                    )
                    logger.info(f"✅ Payment checkout created successfully")
                    logger.info(f"  - Payment URL: {response.get('payment_url', 'N/A')}")
                    logger.info(f"  - Payment ID: {response.get('id', 'N/A')}")

                    # Update booking status
                    booking.status = 'pending'
                    booking.save()
                    logger.info(f"💾 Updated booking {booking.booking_reference} status to pending")

                    # Send payment initiation email for card payments
                    try:
                        # Get passenger details and flight data from booking
                        passenger_details = booking.passenger_details or {}
                        flight_data = booking.flight_data or {}

                        # Extract flight information
                        flight_title = flight_data.get('title', flight_data.get('flight_number', 'Flight'))
                        flight_segments = flight_data.get('flight_segments', [{}])
                        first_segment = flight_segments[0] if flight_segments else {}

                        departure_airport = first_segment.get('departure_airport', flight_data.get('origin', 'NBO'))
                        arrival_airport = first_segment.get('arrival_airport', flight_data.get('destination', 'LHR'))
                        departure_time = first_segment.get('departure_time', flight_data.get('departure_time', 'TBD'))
                        arrival_time = first_segment.get('arrival_time', flight_data.get('arrival_time', 'TBD'))
                        airline = first_segment.get('operating_airline', flight_data.get('airline', 'Airline'))
                        aircraft = first_segment.get('aircraft_type', 'Aircraft')
                        duration = flight_data.get('duration', 'Duration')

                        initiation_html = f"""
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <meta charset="utf-8">
                            <title>Payment Initiated - Bodrless</title>
                            <style>
                                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); }}
                                .container {{ max-width: 700px; margin: 0 auto; background: white; border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.15); overflow: hidden; }}
                                .header {{ background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 50%, #1e40af 100%); color: white; padding: 25px; text-align: center; position: relative; }}
                                .header::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="dots" width="20" height="20" patternUnits="userSpaceOnUse"><circle cx="10" cy="10" r="1" fill="rgba(255,255,255,0.1)"/></pattern></defs><rect width="100" height="100" fill="url(%23dots)"/></svg>'); pointer-events: none; }}
                                .header h1 {{ margin: 0; font-size: 26px; font-weight: 700; text-shadow: 0 2px 4px rgba(0,0,0,0.2); }}
                                .header p {{ margin: 8px 0 0 0; font-size: 16px; opacity: 0.9; }}
                                .content {{ padding: 30px; }}
                                .flight-card {{ background: linear-gradient(135deg, #f8fafc, #f1f5f9); border-radius: 12px; padding: 20px; margin: 20px 0; border-left: 4px solid #3b82f6; }}
                                .flight-header {{ background: #3b82f6; color: white; padding: 15px; border-radius: 10px; margin-bottom: 20px; text-align: center; }}
                                .flight-info {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 15px 0; }}
                                .info-section {{ margin-bottom: 15px; }}
                                .info-label {{ font-weight: 600; color: #475569; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }}
                                .info-value {{ color: #1e293b; font-size: 16px; font-weight: 500; }}
                                .route {{ font-size: 18px; font-weight: 700; color: #3b82f6; text-align: center; margin: 15px 0; }}
                                .payment-info {{ background: linear-gradient(135deg, #eff6ff, #dbeafe); padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 4px solid #f59e0b; }}
                                .status {{ text-align: center; margin: 20px 0; }}
                                .status-badge {{ background: #f59e0b; color: white; padding: 8px 16px; border-radius: 20px; font-weight: 600; text-transform: uppercase; font-size: 14px; }}
                                .footer {{ text-align: center; color: #64748b; font-size: 13px; margin-top: 30px; padding: 20px; background: #f8fafc; border-radius: 0 0 15px 15px; }}
                                .highlight {{ background: linear-gradient(135deg, #fef3c7, #fde68a); padding: 15px; border-radius: 10px; border-left: 4px solid #f59e0b; margin: 20px 0; }}
                            </style>
                        </head>
                        <body>
                            <div class="container">
                                <div class="header">
                                    <h1>💳 Payment Initiated</h1>
                                    <p>Your booking is being processed</p>
                                </div>

                                <div class="content">
                                    <div class="flight-card">
                                        <div class="flight-header">
                                            <h3 style="margin: 0; color: white;">✈️ Flight Information</h3>
                                        </div>

                                        <div style="text-align: center; margin-bottom: 20px;">
                                            <div class="route">
                                                {departure_airport.split(' (')[0]} → {arrival_airport.split(' (')[0]}
                                            </div>
                                            <p style="color: #64748b; margin: 5px 0 0 0; font-size: 14px;">{flight_title} • {duration}</p>
                                        </div>

                                        <div class="flight-info">
                                            <div>
                                                <div class="info-section">
                                                    <div class="info-label">Departure</div>
                                                    <div class="info-value">{departure_airport}</div>
                                                    <div style="color: #64748b; font-size: 14px;">{departure_time}</div>
                                                </div>
                                                <div class="info-section">
                                                    <div class="info-label">Airline</div>
                                                    <div class="info-value">{airline}</div>
                                                    <div style="color: #64748b; font-size: 14px;">{aircraft}</div>
                                                </div>
                                            </div>
                                            <div>
                                                <div class="info-section">
                                                    <div class="info-label">Arrival</div>
                                                    <div class="info-value">{arrival_airport}</div>
                                                    <div style="color: #64748b; font-size: 14px;">{arrival_time}</div>
                                                </div>
                                                <div class="info-section">
                                                    <div class="info-label">Duration</div>
                                                    <div class="info-value">{duration}</div>
                                                    <div style="color: #64748b; font-size: 14px;">Direct Flight</div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    <div class="payment-info">
                                        <h3 style="margin: 0 0 15px 0; color: #1e293b;">💳 Payment Details</h3>
                                        <div class="flight-info">
                                            <div>
                                                <div class="info-section">
                                                    <div class="info-label">Booking Reference</div>
                                                    <div class="info-value" style="font-family: monospace; color: #3b82f6;">{booking.booking_reference}</div>
                                                </div>
                                                <div class="info-section">
                                                    <div class="info-label">Amount</div>
                                                    <div class="info-value" style="font-size: 18px; color: #059669; font-weight: 600;">{currency} {amount:,}</div>
                                                </div>
                                            </div>
                                            <div>
                                                <div class="info-section">
                                                    <div class="info-label">Payment Method</div>
                                                    <div class="info-value">Card Payment</div>
                                                </div>
                                                <div class="info-section">
                                                    <div class="info-label">Status</div>
                                                    <div class="status">
                                                        <span class="status-badge">Pending</span>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    <div class="highlight">
                                        <h4 style="margin: 0 0 10px 0; color: #92400e;">💳 Next Steps</h4>
                                        <p style="margin: 5px 0; color: #78350f;">• You will be redirected to complete your card payment</p>
                                        <p style="margin: 5px 0; color: #78350f;">• Follow the payment instructions on the next page</p>
                                        <p style="margin: 5px 0; color: #78350f;">• You will receive your e-ticket once payment is confirmed</p>
                                    </div>
                                </div>

                                <div class="footer">
                                    <p>Thank you for choosing Bodrless for your journey!</p>
                                    <p>For support: support@bodrless.com | +254 700 000 000</p>
                                    <p style="margin-top: 15px; opacity: 0.8; font-size: 12px;">This is a payment confirmation. Your e-ticket will be sent separately upon payment completion.</p>
                                </div>
                            </div>
                        </body>
                        </html>
                        """

                        send_mail(
                            subject=f'💳 Payment Initiated - Flight {flight_title} ({booking.booking_reference})',
                            message=f'''
Dear {passenger_details.get('full_name', passenger_details.get('first_name', 'Valued Customer'))},

Your payment has been initiated for your flight booking!

FLIGHT DETAILS:
- Flight: {flight_title}
- Route: {departure_airport.split(' (')[0]} → {arrival_airport.split(' (')[0]}
- Departure: {departure_time}
- Arrival: {arrival_time}
- Airline: {airline}
- Duration: {duration}

BOOKING DETAILS:
- Booking Reference: {booking.booking_reference}
- Amount: {currency} {amount:,}
- Payment Method: Card Payment
- Status: Pending

You will be redirected to complete your card payment.

You will receive your beautiful e-ticket email once the payment is successfully processed.

Thank you for choosing Bodrless!
The Bodrless Team
''',
                            from_email=None,
                            recipient_list=[email],
                            fail_silently=True,
                            html_message=initiation_html
                        )
                        logger.info(f"Payment initiation email sent to {email}")
                    except Exception as e:
                        logger.warning(f"Failed to send payment initiation email: {e}")

                    return Response({
                        'success': True,
                        'payment_url': response.get('payment_url', ''),
                        'payment_id': response.get('id', ''),
                        'message': 'Payment initiated successfully'
                    })
                except Exception as checkout_error:
                    logger.error(f"❌ Payment checkout failed: {checkout_error}")
                    logger.error(f"  - Error type: {type(checkout_error).__name__}")
                    import traceback
                    logger.error(f"  - Traceback: {traceback.format_exc()}")

                    # Return proper error response instead of None
                    return Response({
                        'error': 'Payment gateway temporarily unavailable',
                        'details': 'The payment service is currently experiencing issues. Please try again later or contact support.',
                        'suggestion': 'You can try again in a few minutes or use an alternative payment method.',
                        'fallback_available': True
                    }, status=503)

            except Exception as sdk_error:
                logger.error(f"IntaSend SDK error: {sdk_error}")
                logger.error(f"Error type: {type(sdk_error)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")

                # Check if it's an authentication error
                if 'authentication_failed' in str(sdk_error) or 'Session expired' in str(sdk_error):
                    return Response({
                        'error': 'Payment gateway authentication failed',
                        'details': 'Invalid or expired IntaSend API key. Please check your configuration.',
                        'suggestion': 'Use a valid IntaSend API key from https://sandbox.intasend.com',
                        'fallback_available': True
                    }, status=503)
                else:
                    return Response({
                        'error': 'Payment gateway error',
                        'details': str(sdk_error),
                        'fallback_available': True
                    }, status=503)

        except Exception as e:
            logger.error(f"IntaSend payment error: {e}")
            return Response({'error': 'Failed to initiate payment'}, status=500)


@method_decorator(csrf_exempt, name='dispatch')
class PaymentCallbackView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        try:
            # Handle IntaSend payment callback
            callback_data = json.loads(request.body)
            payment_id = callback_data.get('payment_id')
            status = callback_data.get('status')

            logger.info(f"Payment callback received: {payment_id} - {status}")

            # Update booking status based on payment result
            if status == 'COMPLETED':
                # Find booking by payment reference
                booking = FlightBooking.objects.filter(
                    booking_reference=callback_data.get('reference', '')
                ).first()

                if booking:
                    booking.status = 'confirmed'
                    booking.save()

                    # Send beautiful e-ticket email
                    try:
                        # Prepare passenger details for email
                        passenger_details = booking.passenger_details or {}
                        flight_data = booking.flight_data or {}

                        # Create enhanced flight details for email
                        flight_details = {
                            'title': f"{flight_data.get('airline', 'Flight')} {flight_data.get('flight_number', 'N/A')}",
                            'price': f"{booking.currency} {booking.total_amount}",
                            'duration': flight_data.get('duration', 'N/A'),
                            'detailed_info': {
                                'route': f"{flight_data.get('origin', 'NBO')} → {flight_data.get('destination', 'LHR')}",
                                'flight_segments': flight_data.get('flight_segments', []),
                                'departure_date': datetime.now().strftime('%Y-%m-%d'),
                                'terminal_info': flight_data.get('terminal_info', []),
                                'stops_info': flight_data.get('stops_info', []),
                                'price_breakdown': flight_data.get('price_breakdown', []),
                                'baggage_info': flight_data.get('baggage_info', []),
                                'services': flight_data.get('services', {})
                            }
                        }

                        # Create the beautiful HTML email
                        html_message = f"""
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <meta charset="utf-8">
                            <title>Bodrless E-Ticket</title>
                            <style>
                                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); }}
                                .ticket-container {{ max-width: 650px; margin: 0 auto; background: white; border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.15); overflow: hidden; }}
                                .ticket-header {{ background: linear-gradient(135deg, #f59e0b 0%, #f97316 50%, #ea580c 100%); color: white; padding: 25px; text-align: center; position: relative; }}
                                .ticket-header::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="75" cy="75" r="1" fill="rgba(255,255,255,0.05)"/><circle cx="50" cy="10" r="0.5" fill="rgba(255,255,255,0.08)"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>'); pointer-events: none; }}
                                .ticket-header h1 {{ margin: 0; font-size: 28px; font-weight: 700; text-shadow: 0 2px 4px rgba(0,0,0,0.2); }}
                                .ticket-header p {{ margin: 8px 0 0 0; font-size: 16px; opacity: 0.9; }}
                                .ticket-body {{ padding: 30px; }}
                                .ticket-section {{ margin-bottom: 25px; background: #f8fafc; border-radius: 12px; padding: 20px; border-left: 4px solid #f59e0b; }}
                                .ticket-section h3 {{ margin: 0 0 15px 0; color: #1e293b; font-size: 18px; font-weight: 600; }}
                                .info-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
                                .info-item {{ margin-bottom: 12px; }}
                                .info-label {{ font-weight: 600; color: #475569; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }}
                                .info-value {{ color: #1e293b; font-size: 16px; font-weight: 500; }}
                                .flight-route {{ background: linear-gradient(135deg, #f59e0b, #f97316); color: white; padding: 15px; border-radius: 10px; margin: 15px 0; text-align: center; }}
                                .flight-route .route {{ font-size: 20px; font-weight: 700; margin-bottom: 5px; }}
                                .flight-route .details {{ font-size: 14px; opacity: 0.9; }}
                                .qr-section {{ text-align: center; margin: 25px 0; }}
                                .qr-code {{ width: 120px; height: 120px; background: linear-gradient(45deg, #f1f5f9, #e2e8f0); border: 3px dashed #cbd5e1; border-radius: 10px; display: inline-flex; align-items: center; justify-content: center; color: #64748b; font-size: 28px; font-weight: bold; margin: 10px; }}
                                .important-info {{ background: linear-gradient(135deg, #fef3c7, #fde68a); padding: 20px; border-radius: 10px; border-left: 4px solid #f59e0b; }}
                                .important-info h4 {{ margin: 0 0 12px 0; color: #92400e; font-size: 16px; }}
                                .important-info p {{ margin: 6px 0; color: #78350f; font-size: 14px; }}
                                .ticket-footer {{ background: #1e293b; color: white; padding: 20px; text-align: center; font-size: 13px; }}
                                .status-badge {{ position: absolute; top: 20px; right: 20px; background: #10b981; color: white; padding: 6px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; text-transform: uppercase; }}
                                .booking-ref {{ background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px; font-family: monospace; font-size: 14px; margin-top: 10px; display: inline-block; }}
                            </style>
                        </head>
                        <body>
                            <div class="ticket-container">
                                <div class="ticket-header">
                                    <div class="status-badge">Confirmed</div>
                                    <h1>🎫 Bodrless E-Ticket</h1>
                                    <p>Your journey awaits!</p>
                                    <div class="booking-ref">Booking Ref: {booking.booking_reference}</div>
                                </div>

                                <div class="ticket-body">
                                    <div class="flight-route">
                                        <div class="route">{flight_details['detailed_info']['route']}</div>
                                        <div class="details">{flight_details['title']} • {flight_details['duration']} • {flight_details['price']}</div>
                                    </div>

                                    <div class="ticket-section">
                                        <h3>👤 Passenger Information</h3>
                                        <div class="info-grid">
                                            <div>
                                                <div class="info-item">
                                                    <div class="info-label">Full Name</div>
                                                    <div class="info-value">{passenger_details.get('full_name', passenger_details.get('first_name', '') + ' ' + passenger_details.get('last_name', ''))}</div>
                                                </div>
                                                <div class="info-item">
                                                    <div class="info-label">Email</div>
                                                    <div class="info-value">{passenger_details.get('email', 'N/A')}</div>
                                                </div>
                                                <div class="info-item">
                                                    <div class="info-label">Phone</div>
                                                    <div class="info-value">{passenger_details.get('phone_number', 'N/A')}</div>
                                                </div>
                                            </div>
                                            <div>
                                                <div class="info-item">
                                                    <div class="info-label">Passport/ID</div>
                                                    <div class="info-value">{passenger_details.get('passport_number', passenger_details.get('id_number', 'N/A'))}</div>
                                                </div>
                                                <div class="info-item">
                                                    <div class="info-label">Booking Date</div>
                                                    <div class="info-value">{datetime.now().strftime('%B %d, %Y')}</div>
                                                </div>
                                                <div class="info-item">
                                                    <div class="info-label">Status</div>
                                                    <div class="info-value" style="color: #10b981; font-weight: 600;">✓ Confirmed</div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    <div class="ticket-section">
                                        <h3>✈️ Flight Details</h3>
                                        <div class="info-grid">
                                            <div>
                                                <div class="info-item">
                                                    <div class="info-label">Airline</div>
                                                    <div class="info-value">{flight_data.get('airline', 'N/A')}</div>
                                                </div>
                                                <div class="info-item">
                                                    <div class="info-label">Flight Number</div>
                                                    <div class="info-value">{flight_data.get('flight_number', 'N/A')}</div>
                                                </div>
                                            </div>
                                            <div>
                                                <div class="info-item">
                                                    <div class="info-label">Departure</div>
                                                    <div class="info-value">{flight_data.get('departure_time', 'TBD')}</div>
                                                </div>
                                                <div class="info-item">
                                                    <div class="info-label">Arrival</div>
                                                    <div class="info-value">{flight_data.get('arrival_time', 'TBD')}</div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    <div class="qr-section">
                                        <div class="qr-code">QR</div>
                                        <p style="color: #64748b; font-size: 14px; margin: 10px 0 0 0;">Show this QR code at the airport check-in counter</p>
                                    </div>

                                    <div class="important-info">
                                        <h4>⚠️ Important Travel Information</h4>
                                        <p>• Please arrive at the airport at least 2 hours before departure</p>
                                        <p>• Bring valid government-issued ID and this e-ticket</p>
                                        <p>• Check-in closes 45 minutes before departure</p>
                                        <p>• Baggage allowance and restrictions apply - check with airline</p>
                                    </div>
                                </div>

                                <div class="ticket-footer">
                                    <p>Thank you for choosing Bodrless for your travel needs!</p>
                                    <p>For support: support@bodrless.com | +254 700 000 000</p>
                                    <p style="margin-top: 15px; opacity: 0.8; font-size: 12px;">This is an electronically generated ticket. No signature required.</p>
                                </div>
                            </div>
                        </body>
                        </html>
                        """

                        # Send the HTML email
                        send_mail(
                            subject=f'🎫 Your Bodrless E-Ticket - {booking.booking_reference}',
                            message=f'''
Dear {passenger_details.get('full_name', passenger_details.get('first_name', 'Valued Customer'))},

Your flight booking has been confirmed! Here are your travel details:

BOOKING REFERENCE: {booking.booking_reference}
PASSENGER: {passenger_details.get('full_name', passenger_details.get('first_name', '') + ' ' + passenger_details.get('last_name', ''))}
FLIGHT: {flight_details['title']}
ROUTE: {flight_details['detailed_info']['route']}
AMOUNT: {flight_details['price']}

Your e-ticket is attached to this email. Please bring a printed copy or show this email at the airport.

For check-in details and baggage information, please visit our website or contact your airline directly.

Safe travels!
The Bodrless Team
''',
                            from_email=None,
                            recipient_list=[passenger_details.get('email', booking.passenger_details.get('email', ''))],
                            fail_silently=False,
                            html_message=html_message
                        )

                        logger.info(f"Beautiful e-ticket email sent to {passenger_details.get('email')} for booking {booking.booking_reference}")
                    except Exception as e:
                        logger.warning(f"Failed to send e-ticket email: {e}")

            return Response({'status': 'received'})

        except Exception as e:
            logger.error(f"Payment callback error: {e}")
            return Response({'error': 'Callback processing failed'}, status=500)


@method_decorator(csrf_exempt, name='dispatch')
class BookingConfirmationView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            data = json.loads(request.body)
            booking_reference = data.get('booking_reference')
            email = data.get('email')
            passenger_details = data.get('passenger_details', {})
            flight_details = data.get('flight_details', {})

            if not all([booking_reference, email, passenger_details, flight_details]):
                return Response({'error': 'Missing required booking information'}, status=400)

            # Create a beautiful HTML email template
            html_message = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>Booking Confirmation - Bodrless</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                    .container {{ max-width: 600px; margin: 0 auto; background-color: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                    .header {{ background: linear-gradient(135deg, #f59e0b, #f97316); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                    .ticket {{ border: 2px solid #f59e0b; border-radius: 15px; margin: 20px; padding: 20px; background: linear-gradient(145deg, #fff, #f9f9f9); }}
                    .flight-info {{ background-color: #f59e0b; color: white; padding: 15px; border-radius: 10px; margin-bottom: 20px; }}
                    .passenger-info {{ background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px; }}
                    .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 30px; padding: 20px; }}
                    .highlight {{ background-color: #fef3c7; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                    .qr-placeholder {{ width: 100px; height: 100px; background-color: #e5e7eb; border: 2px dashed #9ca3af; display: inline-block; text-align: center; line-height: 100px; color: #6b7280; font-size: 24px; margin: 10px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🎫 Bodrless E-Ticket</h1>
                        <p>Your flight booking has been confirmed!</p>
                    </div>

                    <div class="ticket">
                        <div class="flight-info">
                            <h2>Flight Information</h2>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px;">
                                <div>
                                    <strong>Flight:</strong> {flight_details.get('title', 'N/A')}<br>
                                    <strong>Route:</strong> {flight_details.get('detailed_info', {}).get('route', 'N/A')}<br>
                                    <strong>Duration:</strong> {flight_details.get('duration', 'N/A')}
                                </div>
                                <div>
                                    <strong>Price:</strong> {flight_details.get('price', 'N/A')}<br>
                                    <strong>Date:</strong> {flight_details.get('detailed_info', {}).get('departure_date', 'TBD')}<br>
                                    <strong>Status:</strong> <span style="color: #10b981;">Confirmed</span>
                                </div>
                            </div>
                        </div>

                        <div class="passenger-info">
                            <h3>👤 Passenger Details</h3>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 10px;">
                                <div>
                                    <strong>Name:</strong> {passenger_details.get('first_name', '')} {passenger_details.get('last_name', '')}<br>
                                    <strong>Email:</strong> {passenger_details.get('email', '')}<br>
                                    <strong>Phone:</strong> {passenger_details.get('phone_number', '')}
                                </div>
                                <div>
                                    <strong>Passport:</strong> {passenger_details.get('passport_number', 'N/A')}<br>
                                    <strong>ID Number:</strong> {passenger_details.get('id_number', 'N/A')}<br>
                                    <strong>Booking Ref:</strong> {booking_reference}
                                </div>
                            </div>
                        </div>

                        <div class="highlight">
                            <h4>✈️ Flight Segments</h4>
                            {self.get_flight_segments_html(flight_details)}
                        </div>

                        <div style="text-align: center; margin: 20px 0;">
                            <div class="qr-placeholder">QR</div>
                            <p><small>Show this QR code at the airport</small></p>
                        </div>

                        <div style="background-color: #fef3c7; padding: 15px; border-radius: 10px; text-align: center;">
                            <h4 style="margin: 0 0 10px 0; color: #92400e;">Important Information</h4>
                            <p style="margin: 5px 0; color: #78350f;">• Please arrive at the airport 2 hours before departure</p>
                            <p style="margin: 5px 0; color: #78350f;">• Bring valid identification and this e-ticket</p>
                            <p style="margin: 5px 0; color: #78350f;">• Check-in closes 45 minutes before departure</p>
                        </div>
                    </div>

                    <div class="footer">
                        <p>Thank you for choosing Bodrless for your travel needs!</p>
                        <p>For support, contact us at support@bodrless.com</p>
                        <p><small>This is an electronically generated ticket. No signature required.</small></p>
                    </div>
                </div>
            </body>
            </html>
            """

            # Send the HTML email
            try:
                send_mail(
                    subject='🎫 Your Bodrless E-Ticket - Booking Confirmed',
                    message=f'''
Dear {passenger_details.get('first_name', 'Valued Customer')},

Your flight booking has been confirmed! Here are your booking details:

BOOKING REFERENCE: {booking_reference}
PASSENGER: {passenger_details.get('first_name', '')} {passenger_details.get('last_name', '')}
FLIGHT: {flight_details.get('title', 'N/A')}
ROUTE: {flight_details.get('detailed_info', {}).get('route', 'N/A')}
PRICE: {flight_details.get('price', 'N/A')}

Your e-ticket has been attached to this email. Please bring a printed copy or show this email at the airport.

For check-in details and baggage information, please visit our website or contact your airline directly.

Safe travels!
The Bodrless Team
''',
                    from_email=None,
                    recipient_list=[email],
                    fail_silently=False,
                    html_message=html_message
                )

                logger.info(f"Confirmation email sent to {email} for booking {booking_reference}")
                return Response({'message': 'Confirmation email sent successfully'})

            except Exception as email_error:
                logger.error(f"Failed to send confirmation email: {email_error}")
                return Response({'error': 'Booking created but email sending failed'}, status=500)

        except Exception as e:
            logger.error(f"Booking confirmation error: {e}")
            return Response({'error': 'Failed to send confirmation email'}, status=500)

    def get_flight_segments_html(self, flight_details):
        segments = flight_details.get('detailed_info', {}).get('flight_segments', [])
        if not segments:
            return "<p>Flight details will be provided by the airline.</p>"

        html = ""
        for i, segment in enumerate(segments[:3]):  # Show max 3 segments
            html += f"""
            <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin: 5px 0;">
                <strong>{segment.get('flight_number', 'N/A')}</strong><br>
                {segment.get('departure_airport', 'N/A')} → {segment.get('arrival_airport', 'N/A')}<br>
                <small>{segment.get('departure_time', 'N/A')} - {segment.get('arrival_time', 'N/A')}</small>
            </div>
            """
        return html


@method_decorator(csrf_exempt, name='dispatch')
class RootView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        return Response({
            'message': 'Bodrless Django API',
            'version': '1.0.0',
            'status': 'running',
            'endpoints': {
                'auth': {
                    'register': 'POST /api/auth/register/',
                    'login': 'POST /api/auth/login/',
                    'logout': 'POST /api/auth/logout/',
                    'profile': 'GET/PUT /api/auth/profile/'
                },
                'flights': {
                    'search': 'POST /api/flights/search/',
                    'chat': 'POST /api/chat/'
                },
                'booking': {
                    'create': 'POST /api/booking/',
                    'payment_options': 'GET /api/payment/options/',
                    'intasend_payment': 'POST /api/payment/intasend/',
                    'payment_callback': 'POST /api/payment/callback/'
                },
                'health': 'GET /api/health/'
            }
        })

