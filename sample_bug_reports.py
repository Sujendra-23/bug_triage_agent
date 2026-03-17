# Sample Bug Reports for Bug Triage Agent
# =========================================
# Use any of these with:
#   python main.py --report "paste the bug report here"
# Or just run:
#   python main.py
# (uses sample_1 by default)


# ── SAMPLE 1: Redis connection exhaustion ─────────────────────────────────────
SAMPLE_1 = """
API response time increased by 300% after deploying v2.3.1.
Redis connection timeout errors are appearing in logs every 30 seconds.
Affected endpoints: /api/v1/orders, /api/v1/cart
The issue started immediately after the deployment at 14:32 UTC.
CPU usage is normal (30%), but memory usage has been climbing steadily from
400MB to 1.8GB over the past 2 hours.
Error log sample:
  redis.exceptions.ConnectionError: Error 111 connecting to redis:6379. Connection refused.
  ConnectionPool: max connections reached (limit=100)
"""

# ── SAMPLE 2: Kafka consumer lag ──────────────────────────────────────────────
SAMPLE_2 = """
Customers are not receiving order confirmation emails.
The delay is between 30 to 60 minutes from order placement.
This started after a config change pushed at 09:15 UTC today.
Kafka consumer group lag for the notifications topic is growing — currently at 42,000 messages.
The consumer service CPU and memory look normal.
No errors in the consumer logs, messages are being processed but very slowly.
Throughput appears to have dropped from the usual ~4,000 messages/min to around 180/min.
"""

# ── SAMPLE 3: Deployment failure / missing env var ────────────────────────────
SAMPLE_3 = """
Payment endpoint returning 500 Internal Server Error for 100% of requests.
Issue started exactly at the time of v3.0.0 deployment (11:45 UTC).
All other endpoints are working fine.
Error in logs:
  KeyError: 'PAYMENT_GATEWAY_URL'
  File "payments/gateway.py", line 34, in get_gateway_url
    return os.environ['PAYMENT_GATEWAY_URL']
Rollback to v2.9.1 is being considered but want root cause first.
"""

# ── SAMPLE 4: OOM crash in Docker container ───────────────────────────────────
SAMPLE_4 = """
Media processing service containers are being killed and restarting every 20-30 minutes.
kubectl describe pod shows OOMKilled as the reason.
Memory limit for the container is set to 512MB.
Issue began after deploying a new image resizing feature in v4.2.0.
The service handles user-uploaded product photos.
During peak upload hours the restarts are more frequent (every 10 minutes).
Memory usage ramps up quickly from ~100MB to 512MB and then the container is killed.
"""

# ── SAMPLE 5: gRPC deadline exceeded ─────────────────────────────────────────
SAMPLE_5 = """
Inventory service returning DEADLINE_EXCEEDED for approximately 25% of gRPC calls.
Issue is intermittent but gets worse during peak traffic hours (12:00-14:00 UTC).
Server CPU and memory are well within limits.
Network latency between services appears normal (< 5ms).
The issue appeared after refactoring the inventory service client in v1.5.0
to create a new gRPC channel on each incoming request for thread safety.
Error:
  grpc._channel._InactiveRpcError: StatusCode.DEADLINE_EXCEEDED
  Deadline Exceeded after 5000ms
"""
