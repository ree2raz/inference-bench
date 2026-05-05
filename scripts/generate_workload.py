"""Generate workload.jsonl with synthetic prompts in two length regimes.

Short regime: ~200 token prompts (chat-style questions)
Long regime: ~1800 token prompts (RAG-style with context padding)
"""
import json
import random
import os

SHORT_TOPICS = [
    "Explain the difference between a mutex and a semaphore in operating systems.",
    "What is the CAP theorem and why does it matter for distributed databases?",
    "Describe how TCP congestion control works in simple terms.",
    "What are the tradeoffs between B-trees and LSM trees for storage engines?",
    "Explain the actor model of concurrency and where it's used.",
    "What is a memory barrier and why do we need them in concurrent programming?",
    "Describe the difference between optimistic and pessimistic concurrency control.",
    "What is the Chandy-Lamport snapshot algorithm and when is it used?",
    "Explain the Raft consensus protocol in your own words.",
    "How does a Bloom filter work and what are its limitations?",
    "What is the difference between strong and eventual consistency?",
    "Explain how virtual memory and page tables work together.",
    "Describe the challenges of distributed garbage collection.",
    "What is a content-addressable storage system and where is it useful?",
    "Explain the difference between epoll and select for I/O multiplexing.",
    "What are copy-on-write semantics and where are they used?",
    "Describe how a JIT compiler optimizes code at runtime.",
    "What is the difference between stackful and stackless coroutines?",
    "Explain how a B+ tree index works in a relational database.",
    "What is a vector clock and how does it help with distributed ordering?",
    "Describe the anatomy of a modern NVMe SSD and its performance characteristics.",
    "What is structured concurrency and why is it gaining traction?",
    "Explain how certificate pinning improves TLS security.",
    "Describe the write-ahead logging strategy in database systems.",
    "What is the difference between a microkernel and a monolithic kernel?",
    "Explain how the Kalman filter works and where it's applied.",
    "What are the implications of the Amdahl's law for parallel computing?",
    "Describe how a write-back cache differs from a write-through cache.",
    "What is two-phase commit and its failure modes?",
    "Explain the difference between deadlock, livelock, and starvation.",
    "How does QUIC improve upon TCP for modern web applications?",
    "Describe the working set model and its impact on program performance.",
    "What is a red-black tree and when would you choose it over a hash map?",
    "Explain the hidden terminal problem in wireless networks.",
    "What is speculative execution and what are its security implications?",
    "Describe how consistent hashing enables horizontal scaling.",
    "What are the advantages of columnar storage over row-based storage?",
    "Explain the Paxos protocol and why it's considered hard to implement.",
    "How does a generational garbage collector reduce pause times?",
    "Describe the challenges of implementing distributed transactions.",
    "What is the difference between symmetric and asymmetric encryption?",
    "Explain how a neural network's backpropagation algorithm works at a high level.",
    "Describe what a syscall is and the cost of crossing the user-kernel boundary.",
    "What is a memory-mapped file and what are its advantages?",
    "Explain the purpose of the TLB and what happens on a TLB miss.",
    "Describe the problems with NAT and how IPv6 aims to solve them.",
    "What is an eBPF program and why is it useful for observability?",
    "Explain how the sliding window protocol manages flow control.",
    "Describe the difference between open-addressing and chaining in hash tables.",
]

RAG_CONTEXT_TEMPLATE = """The following is an excerpt from a technical document about {topic}:

{context}

Based on the above document, please answer the following question in detail:
{question}"""

RAG_TOPICS = [
    ("distributed systems", "consensus protocols and their fault tolerance properties"),
    ("machine learning", "transformer architecture and attention mechanisms"),
    ("operating systems", "scheduling algorithms and their fairness guarantees"),
    ("computer networks", "congestion control across different transport protocols"),
    ("databases", "transaction isolation levels and their practical implications"),
    ("programming languages", "type systems and their role in program correctness"),
    ("computer architecture", "cache coherence protocols in multi-core processors"),
    ("security", "authentication protocols and defense against replay attacks"),
    ("cloud computing", "container orchestration and resource allocation strategies"),
    ("data engineering", "stream processing frameworks and exactly-once semantics"),
]

QUESTIONS = [
    "Summarize the key technical challenges described in this document.",
    "What are the main tradeoffs discussed, and which approach would you recommend?",
    "Describe the architecture proposed and its strengths and weaknesses.",
    "What assumptions does this approach make, and where might it break down?",
    "How does this compare to alternative approaches in the same domain?",
    "What practical deployment considerations arise from this design?",
    "Identify the performance bottlenecks described and suggest mitigations.",
    "What scalability limitations exist in the proposed system?",
    "Describe the failure modes and how the system handles them.",
    "What are the open problems and areas for future improvement?",
]


def _generate_rag_context(topic: str, subtopic: str, target_tokens: int) -> str:
    rng = random.Random(hash((topic, subtopic)))
    paragraphs = []
    paragraph_templates = [
        f"In the field of {topic}, {subtopic} represent a fundamental area of research and engineering. "
        f"The evolution of approaches to {subtopic} has been shaped by both theoretical advances "
        f"and practical constraints encountered in production systems.",

        f"Historically, the development of {subtopic} within {topic} has gone through several phases. "
        f"Early approaches focused on simplicity and correctness, often at the expense of performance. "
        f"As systems scaled, these approaches proved inadequate, leading to new designs that balance "
        f"multiple competing concerns including latency, throughput, consistency, and fault tolerance.",

        f"A key insight in modern approaches to {subtopic} is the recognition that no single solution "
        f"works across all deployment scenarios. Instead, practitioners must carefully evaluate their "
        f"specific requirements including workload characteristics, scale, latency requirements, and "
        f"operational complexity before selecting an approach.",

        f"The theoretical foundations of {subtopic} draw from several areas within {topic} and "
        f"adjacent fields. Key results from distributed computing theory establish fundamental limits "
        f"on what can be achieved, while practical engineering innovations continuously push the "
        f"boundaries of what is feasible in real-world deployments.",

        f"Performance evaluation of different approaches to {subtopic} reveals interesting tradeoffs. "
        f"Benchmark results typically show that the optimal choice depends heavily on the specific "
        f"workload pattern, with different approaches excelling under different conditions. This makes "
        f"it essential to characterize workloads accurately before making architectural decisions.",

        f"Implementation considerations for {subtopic} in production {topic} systems include "
        f"observability, debuggability, and graceful degradation. The most theoretically elegant "
        f"solution is not always the most practical choice when considering the full lifecycle of "
        f"a production system, including deployment, monitoring, and incident response.",

        f"Recent advances in {topic} have introduced new perspectives on {subtopic}. Machine learning "
        f"techniques are being applied to adaptively tune system parameters, while new hardware "
        f"capabilities such as persistent memory and programmable network interfaces open up "
        f"design possibilities that were previously impractical.",

        f"The operational aspects of {subtopic} in {topic} are often underestimated during the design "
        f"phase. Issues such as rolling upgrades, configuration management, capacity planning, and "
        f"incident diagnosis require careful consideration and often influence the choice of approach "
        f"as much as pure performance characteristics.",

        f"Security considerations for {subtopic} in {topic} include access control, audit logging, "
        f"and protection against adversarial inputs. As these systems are increasingly exposed to "
        f"untrusted inputs, robustness against malformed or malicious data has become a critical "
        f"design requirement that must be addressed alongside performance and correctness.",

        f"Looking forward, the trajectory of {subtopic} in {topic} suggests continued innovation "
        f"driven by increasing scale, new hardware capabilities, and evolving workload requirements. "
        f"The convergence of techniques from traditionally separate areas is likely to produce "
        f"novel approaches that challenge current assumptions and design patterns.",
    ]
    idx = 0
    current_len = 0
    while current_len < target_tokens:
        p = paragraph_templates[idx % len(paragraph_templates)]
        paragraphs.append(p)
        current_len += len(p.split())
        idx += 1
    return "\n\n".join(paragraphs)


def generate_workload(output_path: str, num_per_regime: int = 100):
    rng = random.Random(42)
    prompts = []

    for i in range(num_per_regime):
        topic = SHORT_TOPICS[i % len(SHORT_TOPICS)]
        prompts.append({
            "regime": "short",
            "messages": [{"role": "user", "content": topic}],
            "max_tokens": 128,
        })

    for i in range(num_per_regime):
        topic_info = RAG_TOPICS[i % len(RAG_TOPICS)]
        topic, subtopic = topic_info
        question = QUESTIONS[i % len(QUESTIONS)]
        context = _generate_rag_context(topic, subtopic, 1200)
        content = RAG_CONTEXT_TEMPLATE.format(
            topic=topic, context=context, question=question
        )
        prompts.append({
            "regime": "long",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 512,
        })

    rng.shuffle(prompts)

    with open(output_path, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")

    short_count = sum(1 for p in prompts if p["regime"] == "short")
    long_count = sum(1 for p in prompts if p["regime"] == "long")
    print(f"Generated {len(prompts)} prompts ({short_count} short, {long_count} long) → {output_path}")


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "..", "prompts", "workload.jsonl")
    generate_workload(os.path.abspath(out))
