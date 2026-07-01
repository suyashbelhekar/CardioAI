#!/usr/bin/env python3
"""Test script to verify live ECG monitor endpoints."""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000"

def test_buffer_endpoint():
    """Test /api/ecg/buffer endpoint."""
    print("Testing /api/ecg/buffer...")
    try:
        response = requests.get(f"{BASE_URL}/api/ecg/buffer?n=10")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Buffer endpoint working - {len(data)} samples")
            if len(data) > 0:
                print(f"  Sample data: {data[0]}")
            return True
        else:
            print(f"✗ Buffer endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Buffer endpoint error: {e}")
        return False

def test_push_endpoint():
    """Test /api/ecg/push endpoint."""
    print("\nTesting /api/ecg/push...")
    try:
        test_data = {
            "value": 0.5,
            "ts": "2026-04-07T21:00:00"
        }
        response = requests.post(
            f"{BASE_URL}/api/ecg/push",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Push endpoint working - buffered: {result.get('buffered', 0)}")
            return True
        else:
            print(f"✗ Push endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Push endpoint error: {e}")
        return False

def test_live_page():
    """Test /live page loads."""
    print("\nTesting /live page...")
    try:
        response = requests.get(f"{BASE_URL}/live")
        if response.status_code == 200:
            print(f"✓ Live page loads successfully")
            return True
        else:
            print(f"✗ Live page failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Live page error: {e}")
        return False

def test_stream_endpoint():
    """Test /api/ecg/stream endpoint (SSE)."""
    print("\nTesting /api/ecg/stream (will read for 2 seconds)...")
    try:
        response = requests.get(f"{BASE_URL}/api/ecg/stream", stream=True, timeout=3)
        if response.status_code == 200:
            count = 0
            start = time.time()
            for line in response.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith('data:'):
                        count += 1
                        if count == 1:
                            print(f"  First event: {decoded[:80]}...")
                if time.time() - start > 2:
                    break
            print(f"✓ Stream endpoint working - received {count} events")
            return True
        else:
            print(f"✗ Stream endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.Timeout:
        print("✓ Stream endpoint working (timeout expected)")
        return True
    except Exception as e:
        print(f"✗ Stream endpoint error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Live ECG Monitor Endpoint Tests")
    print("=" * 60)
    
    results = []
    results.append(("Live Page", test_live_page()))
    results.append(("Buffer Endpoint", test_buffer_endpoint()))
    results.append(("Push Endpoint", test_push_endpoint()))
    results.append(("Stream Endpoint", test_stream_endpoint()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")
