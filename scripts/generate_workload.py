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


REASONING_TOPICS = [
    "A farmer has a wolf, a goat, and a cabbage that he must transport across a river using a boat that can hold only the farmer and one item. If left unattended together, the wolf eats the goat, or the goat eats the cabbage. How can the farmer transport everything safely?",
    "Prove that the square root of 2 is irrational. Walk through each step of the proof carefully, explaining why each inference is valid.",
    "Three boxes are labeled 'Apples', 'Oranges', and 'Mixed'. All labels are wrong. You can pick one fruit from one box. How do you determine the correct labels for all three boxes?",
    "A ball is dropped from a height of 100 meters. Each bounce reaches 60% of the previous height. Derive the total distance traveled and explain the mathematical series involved.",
    "You have 12 identical-looking coins, one of which is counterfeit (either heavier or lighter). Using a balance scale, what is the minimum number of weighings to identify the counterfeit coin and determine if it's heavier or lighter? Describe the complete algorithm.",
    "A man has to get a fox, a chicken, and a sack of corn across a river. He has a rowboat that can carry only him and one other thing. If the fox and chicken are left together, the fox eats the chicken. If the chicken and corn are left together, the chicken eats the corn. How does the man do it?",
    "Solve this logic puzzle: Five houses in a row are each painted a different color. In each house lives a person with a different nationality, pet, drink, and cigarette brand. The Brit lives in the red house, the Swede keeps dogs, the Dane drinks tea, the green house is just left of the white one, the green house's owner drinks coffee, the Pall Mall smoker keeps birds, the yellow house's owner smokes Dunhill, the center house's owner drinks milk, the Norwegian lives in the first house, the Blend smoker lives next to the cat owner, the horse owner lives next to the Dunhill smoker, the Blue Master smoker drinks beer, the German smokes Prince, the Norwegian lives next to the blue house, the Blend smoker has a neighbor who drinks water. Who owns the fish?",
    "Consider a recursive function f(n) defined as: f(0) = 1, f(1) = 1, f(n) = f(n-1) + f(n-2) for n >= 2. Prove by induction that f(n) < 2^n for all n >= 0.",
    "You are given two eggs and access to a 100-story building. You need to determine the highest floor from which an egg can be dropped without breaking. What is the minimum number of drops needed in the worst case? Explain your strategy.",
    "Prove that any map can be colored with at most four colors such that no two adjacent regions share the same color. While the full proof is extremely complex, explain the key ideas and why this theorem is difficult to prove.",
    "You have a 3-gallon jug and a 5-gallon jug. How do you measure exactly 4 gallons of water? Show all possible solution paths and explain your reasoning.",
    "A bat and ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Most people intuitively answer incorrectly. Explain why the intuitive answer is wrong and derive the correct answer.",
    "There are 100 prisoners about to be executed. The warden offers them a chance: a room with a light bulb and a switch. Each day, one prisoner is taken to the room. They can toggle the switch or leave it. Before the first visit, they can agree on a strategy. How can one prisoner eventually declare with certainty that all prisoners have visited the room?",
    "You have a cake and need to divide it into 8 equal pieces with exactly 3 cuts. Each cut must be a straight line. How do you do it? Think about dimensionality.",
    "Given an array of integers where every element appears twice except for one, find the unique element. Explain the XOR-based solution and prove why it works using properties of XOR.",
    "There are 25 horses and a race track that can race 5 horses at a time. You have no stopwatch. What is the minimum number of races needed to find the top 3 fastest horses? Walk through the complete strategy.",
    "You are in a room with two doors. One leads to freedom, the other to death. There are two guards, one at each door. One always tells the truth, the other always lies. You can ask one question to one guard. What question do you ask to find the door to freedom?",
    "Prove that there are infinitely many prime numbers. Walk through Euclid's proof step by step and explain why the contradiction argument works.",
    "You have a balance scale and 8 balls. One ball is heavier than the others. What is the minimum number of weighings to find the heavy ball? Generalize to n balls.",
    "Consider a round-robin tournament with n players where every player plays every other player exactly once. Prove that the sum of all wins equals the sum of all losses. Then show that it's always possible to order the players such that each player beat the next one in the ordering.",
    "A king decides his kingdom has too many men. He decrees that families must stop having children after their first daughter. What will the eventual gender ratio be? Walk through the probabilistic reasoning carefully.",
    "You are given a sorted array that has been rotated an unknown number of times. Write an algorithm to find a target element in O(log n) time. Explain why binary search still works and handle the edge cases.",
    "Two trains are 100 miles apart, traveling toward each other at 50 mph each. A fly starts at the front of one train and flies at 75 mph to the other train, then turns around and repeats. How far does the fly travel before the trains collide? Solve it two different ways.",
    "You have n switches and n light bulbs in another room. Each switch controls exactly one bulb, but you don't know which. You can flip switches and then enter the bulb room once. How do you determine which switch controls which bulb? The bulbs are initially off and you can touch them.",
    "Prove that the sum of the first n natural numbers is n(n+1)/2. Use three different proof techniques: induction, pairing, and geometric visualization.",
    "A group of 10 people want to share a secret such that any 6 of them can reconstruct it, but any 5 cannot. Explain Shamir's Secret Sharing scheme and why polynomial interpolation guarantees these properties.",
    "You are given 9 coins, one of which is counterfeit (heavier). Using a balance scale, what is the minimum number of weighings to find the counterfeit? Now solve for the case where you don't know if it's heavier or lighter.",
    "A magical maze has rooms connected by one-way doors. Each room has at least one exit. Prove that there exists a cycle in this maze. Then extend this to prove that in any group of n people, there exist two people who have shaken hands with the same number of others.",
    "You need to write a function that determines if a string of brackets (containing (), [], {}) is balanced. Walk through the algorithm design, explain why a stack is the right data structure, and prove correctness.",
    "You are given a linked list that may contain a cycle. Describe Floyd's cycle detection algorithm (tortoise and hare), explain why it works, and derive the meeting point mathematically.",
]


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

    num_reasoning = 30
    for i in range(num_reasoning):
        topic = REASONING_TOPICS[i % len(REASONING_TOPICS)]
        prompts.append({
            "regime": "reasoning",
            "messages": [{"role": "user", "content": topic}],
            "max_tokens": 8192,
        })

    rng.shuffle(prompts)

    with open(output_path, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")

    short_count = sum(1 for p in prompts if p["regime"] == "short")
    long_count = sum(1 for p in prompts if p["regime"] == "long")
    reasoning_count = sum(1 for p in prompts if p["regime"] == "reasoning")
    print(f"Generated {len(prompts)} prompts ({short_count} short, {long_count} long, {reasoning_count} reasoning) → {output_path}")


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "..", "prompts", "workload.jsonl")
    generate_workload(os.path.abspath(out))
