from django.urls import path
from .views import (
    RegisterView, LoginView, LogoutView, ProfileView,
    ChatView, FlightSearchView, HealthView, RootView,
    BookingView, FlightBookingView, TrainBookingView, PaymentOptionsView, IntaSendPaymentView, PaymentCallbackView, BookingConfirmationView
)

urlpatterns = [
    # Authentication endpoints
    path('auth/register/', RegisterView.as_view(), name='register'),
    path('auth/login/', LoginView.as_view(), name='login'),
    path('auth/logout/', LogoutView.as_view(), name='logout'),
    path('auth/profile/', ProfileView.as_view(), name='profile'),

    # Flight and chat endpoints
    path('chat/', ChatView.as_view(), name='chat'),
    path('flights/search/', FlightSearchView.as_view(), name='flight-search'),

    # Booking and payment endpoints
    path('booking/', BookingView.as_view(), name='booking'),
    path('booking/flight/', FlightBookingView.as_view(), name='flight-booking'),
    path('booking/train/', TrainBookingView.as_view(), name='train-booking'),
    path('booking/send-confirmation/', BookingConfirmationView.as_view(), name='booking-confirmation'),
    path('payment/options/', PaymentOptionsView.as_view(), name='payment-options'),
    path('payment/intasend/', IntaSendPaymentView.as_view(), name='intasend-payment'),
    path('payment/callback/', PaymentCallbackView.as_view(), name='payment-callback'),

    # Health and info endpoints
    path('health/', HealthView.as_view(), name='health'),
    path('', RootView.as_view(), name='root'),
]
