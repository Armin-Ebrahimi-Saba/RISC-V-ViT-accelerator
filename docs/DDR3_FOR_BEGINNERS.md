# The DDR3 problems, explained from scratch

This is the story of three bugs in the memory system of this project, told
for someone who has never heard of DDR3. No prior knowledge assumed. Every
term is introduced with an everyday picture before it is used.

---

## 1. What DDR3 is, and why it needs help

**DDR3 is the big memory chip on the board.** It holds 512 megabytes — the
25 MB of neural-network weights, plus all the working data the network
produces as it runs. Think of it as a very large warehouse.

Warehouses are big but slow. Fetching one item means walking to the right
aisle, finding the shelf, and walking back — and DDR3 has the same shape:
every access takes tens of clock cycles, and the memory chip only speaks a
complicated protocol with strict timing that the processor cannot deal with
directly.

So between the processor and the warehouse sits a chain of helpers:

```
processor ──► request mux ──► cache ──► prefetcher ──► controller ──► DDR3 chip
```

- **The controller** speaks the chip's protocol. Think of it as the forklift
  driver who knows the warehouse rules.
- **The cache** is a small, fast shelf right next to the processor — 16 kB,
  holding recent items so the next request for the same thing is instant.
  Like keeping the tools you are using on the bench instead of walking back
  to the store room each time.
- **The prefetcher** guesses what you will want next and fetches it early.
- **The request mux** is the front desk: it receives a request and decides
  who handles it — normally the cache, or, if the memory is not ready yet,
  an *error responder* whose job is to say "not available".

All three bugs were in these helpers. The DDR3 chip and its controller were
never at fault.

---

## 2. The one rule everything depends on: the handshake

The processor and the helpers talk over a **bus** — a set of wires with a
simple protocol. Every request needs two things to happen in the same clock
tick:

- the sender says **"I have something for you"** (a signal called `valid`)
- the receiver says **"I can take it now"** (a signal called `ready`)

Only when both are true at once does the request actually move. This is
exactly like handing someone a cup of coffee: you hold it out (`valid`), they
reach for it (`ready`), and the cup changes hands only when both happen. If
you hold it out and they are not reaching, you must **keep holding it** until
they do. You may not put it down, and you may not swap it for a different cup.

Two of the three bugs were violations of this rule. The third was a subtler
kind of mistake: the handshake was perfect, but the wrong cup was handed over.

---

## 3. Bug 1 — the front desk said "yes" for someone who never took the job

### What it looked like

The processor would run for a while, then simply stop. Not crash — stop.
And it could not be woken: the debugger, which normally can freeze the
processor and ask it "where are you?", got no answer. The processor was
running, but frozen inside a single instruction it could never finish.

### What was actually happening

Go back to the front desk. It sits in front of *two* workers: the cache
(does the real work) and the error responder (says "not available" while
the memory is still warming up). Once the memory is ready, the error
responder is told "you are off duty, take nothing".

Here is the bug. The front desk's **"I can take it now"** light was wired to
the *error responder's* switch. The error responder, being off duty and
idle, kept its switch on "ready" all the time — it had nothing to do, so of
course it was "ready". So the light said *yes* even when the cache was busy
and could not take anything.

The processor saw the light, handed over its request, and considered it
delivered. But the error responder was off duty and did not take it. The
cache was busy and did not take it either. **The request fell on the floor.**
Nobody had it, so nobody would ever answer it, and the processor stood there
waiting for an answer that could not come.

A coffee-shop version: two baristas, one on break. The "next customer" sign
is wired to the one on break, who is always free. You order, the sign says
go ahead, the working barista is busy with someone else and did not hear
you, and the one on break is not making drinks. You wait forever.

### Why it was so hard to find

The usual way to find out where a stuck program is stuck is to freeze it
and read its position. But this processor cannot be frozen while it is
waiting for a memory answer — it is not "running a loop" that can be
interrupted, it is halfway through one instruction with nowhere to go. Every
attempt to inspect it came back empty.

The breakthrough was to stop asking the processor and ask the *bus*
instead. A small piece of hardware was added that watches the front desk
and remembers: *what was the last request that came in and was never
answered? Who sent it? How long has it been waiting?* This watcher can be
read through the debugger even while the processor is frozen, because the
debugger uses its own separate path to the bus.

The first reading said: one write, from the accelerator, thirty requests
outstanding, waiting time maxed out. That pointed straight at the front desk.

### The fix

One line: wire the "I can take it now" light to whichever worker is
*actually on duty* — the cache when the memory is ready, the error responder
when it is not.

---

## 4. Bug 3 — the prefetcher handed back the wrong item

(Numbered 3 because it was found third, but it is simpler than bug 2, so it
comes first here.)

### The small shelf has a filing rule

The cache — the small fast shelf — is 16 kB, and the warehouse is 512 MB.
So the shelf cannot hold everything; it holds 512 slots, and every address
in the warehouse is assigned to exactly one slot, by a simple rule: take a
few digits from the middle of the address, and that is your slot number.

This means many different addresses share a slot. Address A and address B,
far apart in the warehouse, can land in the same slot — like two people
whose surnames both start with "M" being assigned the same pigeonhole. When
B arrives, A is thrown out to make room. That is called an **eviction**.

In this project the weights sit at one end of memory and the working data
32 MB away — and by the arithmetic of the slot rule, **they collide in every
single slot**. Bringing in a piece of working data always throws out a piece
of weights, and vice versa. The shelf is constantly churning.

### What it looked like

The program would say things like `tensor 'blk1.qkv.w' not found` — it was
looking up a weight by name in a directory and not finding it. But the
weight was there: reading the raw memory over the debugger showed every byte
correct. The processor's *cached* reads were returning something else.

### What was actually happening

The prefetcher — the helper that guesses what you will want next — keeps a
little table of "things I fetched early". When the processor asks for an
address, the prefetcher checks: *do I already have that?* If yes, it hands
it over immediately instead of going to the warehouse.

Under heavy collision churn, its bookkeeping went wrong: it would hand over
a line it had fetched for *a different address that mapped to the same
slot*. It was like a coat-check attendant who files coats by the first
letter of your name only, and hands you someone else's coat because it was
on the "M" hook.

### How it was found

A test was written that deliberately writes different patterns to two
regions that collide in every slot, then reads them back alternately so
every access evicts the other. 65 of 256 reads came back with the *other*
region's data. Bypassing the prefetcher — wiring the cache straight to the
controller — made the same test pass 256 of 256. That is what pins it on
the prefetcher and not the cache.

### The fix

At first the prefetcher was simply switched off. That cost some speed
(nothing was fetched early), but every read was correct.

Later the real cause was found. The prefetcher keeps a few numbered trays
for lines it has asked the warehouse for. When it decided a tray's order was
no longer wanted, it handed the tray to a new order *while the old delivery
was still on the truck*. The old delivery then arrived, went onto the tray,
and was handed out as the new order. Now a tray is only reused once its
delivery has arrived. With that fix the same test passes 256 of 256 with the
prefetcher switched back on. It has not yet been run on the board.

---

## 5. Bug 2 — the cache sent the warehouse a stale copy

This was the last one found and the hardest, and it is worth going slowly.

### What "write-back" means

When the processor writes a value, the cache does not walk it to the
warehouse right away. It writes it onto the fast shelf and marks that slot
**dirty** — meaning "this is newer than what the warehouse has". Only when
that slot is later needed for something else — an eviction — does the cache
copy the dirty contents back to the warehouse first. This is called
**write-back**, and it is normal and efficient: if you write the same place
ten times, the warehouse only sees the final value once.

Picture a whiteboard next to your desk. You jot notes on it all day; only
when you run out of space and need to erase a section do you first copy
that section into the permanent notebook.

### What it looked like

The accelerator computed a big matrix — 31,104 numbers — and exactly *one*
of them came out as zero instead of the right value. Always exactly one.
Always the same one for a given input. And always, on inspection, the very
last number the accelerator wrote before moving to its next chunk of work.

Everything upstream checked out: the accelerator issued all 31,104 writes and
every one was acknowledged. The memory chip, tested directly, was perfect.
The number was written, acknowledged, and gone.

### What was actually happening

The cache's whiteboard is a piece of memory, and like all such memory it
has a one-tick delay: what you write this tick becomes readable *next*
tick. The cache's designers knew this. When the processor writes a value
and then reads the same slot on the very next tick, the cache has a special
bypass — call it a sticky note — that says "you just wrote this, here is the
value you wrote, do not bother reading the shelf yet". It works, and the
processor always gets the right answer.

But the *eviction* path — the copy-to-the-notebook path — did not use the
sticky note. It read the shelf directly. So:

1. **Tick N:** a value is written into slot 7. Slot 7 is marked dirty.
2. **Tick N+1:** a request for a *different* address that also maps to slot 7
   arrives. Slot 7 must be evicted. The cache reads slot 7 to copy it to the
   warehouse — but the shelf has not updated yet. It reads the *old*
   contents from before tick N, and sends those to the warehouse.

The write from tick N was acknowledged, marked dirty, and then overwritten
in the warehouse by its own stale predecessor. Gone, with no error anywhere.

Back to the whiteboard: you write a number, and in the same breath someone
says "erase that section, I need it". You reach for the notebook, glance at
the board — but your eye lands on where the number *was* a moment ago, not
where your pen just left it. You copy the old number. Your new one is lost.

### Why it was the accelerator's last write every time

The accelerator works in chunks. The very last write of one chunk is
followed, one tick later, by the first read of the next chunk — and by the
collision arithmetic above, that read often lands in the same slot. Write,
then evict, back to back. The middle writes of a chunk never had that
follow-on read, so they were safe.

### How it was found — and a mistake along the way

A first test seemed to show the memory itself losing 111 words out of
65,536. That turned out to be a bug in the *test*: the random addresses it
picked repeated 111 times, and the checker compared each read against the
value that op wrote rather than the *last* value written to that address.
Exactly 111 repeats, exactly 111 "failures". The memory was fine. This is
recorded because it cost real time and the lesson is general: a memory test
that does not handle repeated addresses is not a memory test.

The real reproduction came from making a simulation test drive the bus the
way a processor does — issuing the next request the *instant* the previous
one is accepted, rather than politely waiting for each answer. Only that
back-to-back rhythm opens the one-tick window. The moment the test did that,
one word in 1,024 came back zero, on the unfixed design, every time.

### The fix

One line: the eviction path reads through the sticky note like everyone
else. After that: 256 of 256 in the test, and on the board every one of
fifteen images produced a depth map identical to the reference, all 15,876
numbers each.

---

## 6. What ties the three together

| | The rule broken | The everyday picture |
|---|---|---|
| Bug 1 | said "ready" on behalf of someone who was not going to act | the "next customer" sign wired to the barista on break |
| Bug 3 | filed by slot alone, forgot whose item it was | the coat-check hook with two people's coats |
| Bug 2 | copied from the shelf before the ink was dry | erasing the whiteboard section you just wrote on |

None of them corrupted data randomly, and none of them raised an error.
Each one produced a *plausible-looking wrong outcome* — a hang with no
message, a "file not found" for a file that existed, a single zero in a sea
of correct numbers. That is what made them slow to find: the system never
said anything was wrong. It just quietly did the wrong thing.

The tools that found them were, in every case, ways of *watching the bus
itself* rather than asking the processor what it thought was happening: a
hardware register that remembers unanswered requests, and simulation tests
that recreate exactly the traffic pattern the bug needs. The full technical
account is in `DEBUGGING.md`; the cycle-by-cycle picture of bug 2 is in
`DATAFLOW.md` §3.
