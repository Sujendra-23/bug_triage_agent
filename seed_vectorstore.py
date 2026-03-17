"""
seed_vectorstore.py — Run this once before using the agent.

Populates ChromaDB with realistic past engineering incidents so the
RAG retrieval has meaningful context to work with.
"""

from agent.vectorstore import add_incidents

PAST_INCIDENTS = [
    {
        "id": "INC-001",
        "text": """Incident: Redis connection pool exhaustion causing API timeouts.
Symptoms: API response time increased 400%, Redis connection timeout errors in logs, 
connection pool size maxed at 100.
Root Cause: A new background job added in v2.1.0 was opening Redis connections without 
releasing them. Each job iteration leaked one connection until the pool was exhausted.
Resolution: Fixed connection leak by using context managers (with redis.client() as r).
Added connection pool monitoring alert when usage exceeds 80%.
Prevention: Code review checklist now includes connection lifecycle verification.
Tags: redis, connection-pool, memory-leak, background-job""",
        "metadata": {"severity": "P1", "service": "api", "version": "2.1.0"},
    },
    {
        "id": "INC-002",
        "text": """Incident: Database query latency spike after schema migration.
Symptoms: 300% increase in p99 query latency after deploying v1.8.0, specific 
endpoints /orders and /invoices affected.
Root Cause: Migration added a new column but forgot to add a corresponding index. 
Full table scans on orders table (8M rows) triggered on every request.
Resolution: Added composite index on (user_id, created_at, status). Query latency 
returned to baseline within 2 minutes of index creation.
Prevention: Migration review process now requires EXPLAIN ANALYZE on affected queries.
Tags: database, index, migration, query-performance""",
        "metadata": {"severity": "P1", "service": "orders-api", "version": "1.8.0"},
    },
    {
        "id": "INC-003",
        "text": """Incident: Memory leak in Python Django worker causing OOM crashes.
Symptoms: Worker processes restarting every 4-6 hours, memory usage growing linearly 
from 200MB to 2GB before crash. Seen after deploying v3.2.1.
Root Cause: Django ORM queryset was being accumulated in a module-level list inside 
a signal handler. Querysets hold references to model instances, preventing garbage collection.
Resolution: Cleared the accumulator list after each batch. Used .values() instead of 
full ORM objects to reduce memory footprint.
Prevention: Added memory usage metrics per worker. Alert on >500MB per process.
Tags: python, django, memory-leak, queryset, signal-handler""",
        "metadata": {"severity": "P2", "service": "worker", "version": "3.2.1"},
    },
    {
        "id": "INC-004",
        "text": """Incident: Kafka consumer lag causing delayed order notifications.
Symptoms: Customers not receiving order confirmation emails for up to 45 minutes.
Kafka consumer group lag growing from 0 to 50,000 messages over 2 hours.
Root Cause: A downstream HTTP call inside the consumer had its timeout increased from 
2s to 30s in a config change. With 10 partitions and 30s per message, throughput 
dropped from 5000/min to 200/min.
Resolution: Moved the HTTP call to a separate async queue. Consumer now just enqueues 
the work. Throughput restored to 5000/min within 10 minutes.
Prevention: Consumer throughput dashboards added. Timeout configuration changes 
require performance impact assessment.
Tags: kafka, consumer-lag, async, timeout, notifications""",
        "metadata": {"severity": "P2", "service": "notifications", "version": "2.4.0"},
    },
    {
        "id": "INC-005",
        "text": """Incident: AWS Lambda cold start latency causing checkout failures.
Symptoms: 15% of checkout requests timing out after deploying new Lambda function. 
p99 latency jumped from 200ms to 12s. Only affects first requests after quiet periods.
Root Cause: New Lambda loaded a 180MB ML model on cold start. VPC attachment added 
another 8-10s. Combined cold start time exceeded the 15s API Gateway timeout.
Resolution: Implemented Lambda provisioned concurrency to keep 5 instances warm. 
Moved model loading to a shared EFS layer loaded once at container startup.
Prevention: Cold start benchmarks added to CI pipeline. Lambda size limit alert at 100MB.
Tags: aws-lambda, cold-start, timeout, ml-model, vpc""",
        "metadata": {"severity": "P1", "service": "checkout", "version": "4.1.0"},
    },
    {
        "id": "INC-006",
        "text": """Incident: gRPC service returning DEADLINE_EXCEEDED on high load.
Symptoms: gRPC inventory service returning DEADLINE_EXCEEDED errors for 20% of 
requests during peak hours. CPU usage normal. Network latency normal.
Root Cause: gRPC channel was created per-request instead of being shared. Each new 
channel triggering TLS handshake added ~200ms overhead. At peak load this caused 
cascading deadline exceeded errors.
Resolution: Implemented gRPC channel pool singleton. Channels are now created once 
at startup and reused across requests.
Prevention: gRPC channel lifecycle reviewed in architecture checklist.
Tags: grpc, channel-pool, tls, deadline, connection-reuse""",
        "metadata": {"severity": "P1", "service": "inventory", "version": "1.5.0"},
    },
    {
        "id": "INC-007",
        "text": """Incident: N+1 query problem after adding product recommendations feature.
Symptoms: /product-detail endpoint went from 50ms to 2100ms after v2.7.0 deploy.
Each page load triggering 40-60 individual SQL queries instead of 2.
Root Cause: New recommendations feature loaded related products in a loop without 
using select_related(). Django ORM issued one query per recommendation (N+1 problem).
Resolution: Added select_related('category', 'brand') and prefetch_related('tags').
Query count dropped from 60 to 3. Response time back to 55ms.
Prevention: Django Debug Toolbar added to staging. Any endpoint with >10 queries 
flagged for review before merge.
Tags: django, orm, n-plus-one, query-optimization, select-related""",
        "metadata": {"severity": "P2", "service": "product-api", "version": "2.7.0"},
    },
    {
        "id": "INC-008",
        "text": """Incident: CI/CD pipeline deploying bad build causing production rollback.
Symptoms: 500 errors on /payment endpoint immediately after deployment of v3.0.0.
New environment variable PAYMENT_GATEWAY_URL not set in production secrets.
Root Cause: New required environment variable added in code but not added to 
deployment runbook or secret manager. Build passed CI because tests mocked the 
external call. Production hit the real code path.
Resolution: Rolled back to v2.9.1 within 4 minutes. Added PAYMENT_GATEWAY_URL to 
secret manager. Added startup validation that checks all required env vars on boot.
Prevention: Environment variable additions now require a corresponding PR to infra repo.
Tags: deployment, environment-variable, rollback, secret-manager, startup-validation""",
        "metadata": {"severity": "P0", "service": "payments", "version": "3.0.0"},
    },
    {
        "id": "INC-009",
        "text": """Incident: MongoDB write throughput degradation after index addition.
Symptoms: Write latency increased 5x after adding a text index to the products 
collection for search feature. Insert throughput dropped from 2000/sec to 400/sec.
Root Cause: Text indexes in MongoDB are expensive to maintain on write. The products 
collection was write-heavy (catalog price updates every 5 minutes). The text index 
was being rebuilt on every price update.
Resolution: Moved text search to a dedicated Elasticsearch index updated asynchronously 
via change streams. Removed text index from MongoDB.
Prevention: Index type and write pattern compatibility review added to schema design checklist.
Tags: mongodb, text-index, write-performance, elasticsearch, change-streams""",
        "metadata": {"severity": "P2", "service": "catalog", "version": "1.9.0"},
    },
    {
        "id": "INC-010",
        "text": """Incident: Docker container OOM kill in production after deploy.
Symptoms: Containers restarting every 30 minutes with OOMKilled status. Memory limit 
set to 512MB but process using 800MB. Started after v4.2.0 deploy.
Root Cause: New image processing feature using Pillow library was loading full 
high-resolution images into memory without streaming. A 20MB JPEG expanded to 
400MB in memory after decoding. Concurrent requests multiplied this.
Resolution: Added image resizing before loading into memory. Used Pillow's 
thumbnail() method with a max dimension cap. Memory usage dropped to 150MB peak.
Prevention: Memory profiling added to load tests. Container memory limit alerts set at 80%.
Tags: docker, oom, pillow, image-processing, memory-limit""",
        "metadata": {"severity": "P1", "service": "media-service", "version": "4.2.0"},
    },
]


if __name__ == "__main__":
    print("Seeding ChromaDB vector store with past incidents...")
    add_incidents(PAST_INCIDENTS)
    print("\nDone. You can now run: python main.py")
