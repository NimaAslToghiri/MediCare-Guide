# Django Backend CSRF Integration Guide for Next.js Frontend

## Overview
This document outlines the Django backend changes implemented to ensure proper CSRF token handling with your Next.js frontend application.

## Backend Changes Summary

### 1. CSRF Cookie Settings (settings.py)
- `CSRF_COOKIE_HTTPONLY = False` - Allows JavaScript to read the CSRF cookie
- `CSRF_COOKIE_SAMESITE = 'Lax'` - Appropriate for development with different ports
- `CSRF_COOKIE_SECURE = False` - Set to True in production with HTTPS
- `CSRF_COOKIE_AGE = 31449600` - 1 year expiration

### 2. New API Endpoints

All endpoints are available under the `/medassist/api/` prefix:

#### CSRF Initialization
- **GET** `/api/initialize-csrf/`
- **Purpose**: Must be called when your Next.js app initializes to set the CSRF cookie
- **Response**: `{"status": "csrf_cookie_set", "message": "CSRF token has been initialized"}`

#### Authentication Endpoints
- **POST** `/medassist/api/login/` - User login
- **POST** `/medassist/api/signup/` - User registration  
- **POST** `/medassist/api/logout/` - User logout
- **GET** `/medassist/api/user-status/` - Check authentication status

#### Chat Endpoints
- **POST** `/medassist/api/chat/` - Send messages/upload files
- **GET** `/medassist/chat/` - Chat page (if needed)

### 3. CSRF Token Requirements

For all POST requests, include the CSRF token via:
- **X-CSRFToken header** (recommended for API calls)
- **csrfmiddlewaretoken form field** (for form submissions)

## Frontend Integration Instructions

### 1. App Initialization
Call the CSRF initialization endpoint when your Next.js app starts:

```javascript
// In your root layout or _app.js
useEffect(() => {
  // Initialize CSRF token on app load
  fetch('/api/initialize-csrf/', {
    method: 'GET',
    credentials: 'include' // Important: include cookies
  }).then(response => response.json())
    .then(data => console.log('CSRF initialized:', data))
    .catch(error => console.error('CSRF initialization failed:', error));
}, []);
```

### 2. CSRF Token Helper Function
Update your existing `getCsrfToken()` function:

```javascript
function getCsrfToken() {
  let cookieValue = null;
  if (document.cookie && document.cookie !== '') {
    const cookies = document.cookie.split(';');
    for (let i = 0; i < cookies.length; i++) {
      const cookie = cookies[i].trim();
      if (cookie.substring(0, 10) === 'csrftoken=') {
        cookieValue = decodeURIComponent(cookie.substring(10));
        break;
      }
    }
  }
  if (!cookieValue) {
    console.warn('CSRF token not found in cookies');
  }
  return cookieValue;
}
```

### 3. API Request Examples

#### Login Request
```javascript
const loginUser = async (username, password) => {
  const formData = new FormData();
  formData.append('username', username);
  formData.append('password', password);
  
  const csrfToken = getCsrfToken();
  if (csrfToken) {
    formData.append('csrfmiddlewaretoken', csrfToken);
  }

  const response = await fetch('/medassist/api/login/', {
    method: 'POST',
    body: formData,
    credentials: 'include'
  });
  
  return response.json();
};
```

#### Chat API Request (multipart/form-data)
```javascript
const sendChatMessage = async (text, file) => {
  const formData = new FormData();
  if (text) formData.append('text_input', text);
  if (file) formData.append('file', file);

  const csrfToken = getCsrfToken();
  
  const response = await fetch('/medassist/api/chat/', {
    method: 'POST',
    body: formData,
    headers: {
      'X-CSRFToken': csrfToken || ''
    },
    credentials: 'include'
  });
  
  return response.json();
};
```

#### Logout Request
```javascript
const logoutUser = async () => {
  const csrfToken = getCsrfToken();
  
  const response = await fetch('/medassist/api/logout/', {
    method: 'POST',
    headers: {
      'X-CSRFToken': csrfToken || '',
      'Content-Type': 'application/x-www-form-urlencoded'
    },
    credentials: 'include'
  });
  
  return response.json();
};
```

### 4. Error Handling

All API endpoints return consistent error responses:

```javascript
{
  "status": "error",
  "message": "Error description"
}
```

Handle CSRF failures (403 errors) by re-initializing the CSRF token:

```javascript
const handleApiCall = async (apiFunction) => {
  try {
    return await apiFunction();
  } catch (error) {
    if (error.status === 403) {
      // CSRF token might be invalid, re-initialize
      await fetch('/api/initialize-csrf/', {
        method: 'GET',
        credentials: 'include'
      });
      // Retry the original request
      return await apiFunction();
    }
    throw error;
  }
};
```

### 5. Next.js Rewrites Configuration

Ensure your `next.config.js` includes proper rewrites for the Django backend:

```javascript
module.exports = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*'
      },
      {
        source: '/medassist/:path*',
        destination: 'http://localhost:8000/medassist/:path*'
      }
    ];
  }
};
```

## Testing the Integration

1. Start your Django server: `python manage.py runserver`
2. Test the CSRF initialization endpoint: `curl http://localhost:8000/api/initialize-csrf/`
3. Verify the `csrftoken` cookie is set in your browser
4. Test authentication endpoints with proper CSRF tokens

## Production Notes

For production deployment:
- Set `CSRF_COOKIE_SECURE = True` in Django settings
- Add your production domain to `CSRF_TRUSTED_ORIGINS`
- Ensure HTTPS is enabled
- Update frontend API URLs to production Django server

## Troubleshooting

**"CSRF token not found" errors:**
- Ensure `/api/initialize-csrf/` is called on app initialization
- Check that `credentials: 'include'` is set in fetch requests
- Verify cookies are enabled in the browser

**403 Forbidden errors:**
- Confirm CSRF token is included in request headers or form data
- Check that the CSRF token cookie exists and is readable by JavaScript
- Ensure the request origin matches Django's allowed hosts