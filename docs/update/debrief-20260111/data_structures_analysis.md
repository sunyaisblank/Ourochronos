# Data Structures in Ourochronos: An Analysis of Presence and Absence

> "Bad programmers worry about the code. Good programmers worry about data structures and their relationships."
> – Linus Torvalds

---

## Introduction: Structure as Thought

Data structures serve as crystallised patterns of thought. When we choose a linked list over an array, we make a statement about how we expect to interact with our data. When we reach for a hash table, we declare that *association* matters more than *sequence*.

Ourochronos, in its current form, provides the barest foundation for structure: contiguous memory and the integer. From these primitives, all structures must be built. This document examines what structures exist, what structures are absent, and what the implications are for temporal computation.

---

## Part I: The Primitive Foundation

### 1.1 The Memory Model

At the foundation lies a simple abstraction:

```
Memory := Address → Value
Address := u16 (0..65535)
Value := u64
```

This gives us 65,536 cells, each holding a 64-bit value. The total addressable space is:

$$\text{Memory Size} = 65536 \times 8 = 524,288 \text{ bytes} = 512 \text{ KB}$$

This represents the memory of constraint rather than that of a modern computer. Within these bounds, all data must live.

### 1.2 The Stack

The stack serves as the primary data structure of Ourochronos, arising from the nature of concatenative computation rather than explicit design.

```ouro
10 20 30    # Stack: [10, 20, 30]
ADD         # Stack: [10, 50]
MUL         # Stack: [500]
```

The stack is:
- **Implicit:** Operations consume and produce stack values without naming
- **Ordered:** LIFO (Last In, First Out) access pattern
- **Homogeneous:** All elements are Value (u64 + Provenance)
- **Unbounded:** Limited only by available memory

The stack primitives form a complete set for manipulation:

| Operation | Effect | Description |
|-----------|--------|-------------|
| `DUP` | `a -- a a` | Duplicate top |
| `DROP` | `a --` | Discard top |
| `SWAP` | `a b -- b a` | Exchange top two |
| `OVER` | `a b -- a b a` | Copy second |
| `ROT` | `a b c -- b c a` | Rotate top three |
| `NIP` | `a b -- b` | Drop second |
| `TUCK` | `a b -- b a b` | Copy top below second |
| `PICK n` | `-- x` | Copy nth element |
| `ROLL n` | `-- x` | Rotate nth to top |

### 1.3 What the Primitives Enable

From stack and memory, we can build:
- **Arrays:** Contiguous memory regions
- **Records:** Fixed-offset field access
- **Linked structures:** Pointer-based traversal
- **Tables:** Computed addressing schemes

We cannot efficiently build:
- **Hash tables:** Without hash functions and collision handling
- **Trees:** Without pointer management abstractions
- **Graphs:** Without edge representation conventions

---

## Part II: Arrays and Contiguous Structures

### 2.1 The Array Convention

An array in Ourochronos is a region of memory with a base address and length:

```
Array := (base_address, length)
```

Representation choices:
1. **Length-prefixed:** Length stored at base_address, data at base_address+1
2. **Descriptor-based:** Separate length and base on stack
3. **Sentinel-terminated:** Special value marks end

The length-prefixed convention:

```ouro
# Create array [10, 20, 30]
100 3 STORE      # Length at address 100
101 10 STORE     # Data[0]
102 20 STORE     # Data[1]
103 30 STORE     # Data[2]

# Access array[1]
PROC ARRAY_GET ( base idx -- val ) {
    ADD 1 ADD FETCH   # base + idx + 1
}
100 1 ARRAY_GET  # Returns 20
```

### 2.2 Array Operations

**Iteration:**
```ouro
PROC ARRAY_EACH ( base quot -- ) {
    # Get length
    OVER FETCH        # base quot len

    # Loop counter
    0                 # base quot len i

    WHILE { 2DUP GT } {
        # Get element: base + i + 1
        3 PICK        # base quot len i base
        OVER          # base quot len i base i
        ADD 1 ADD     # base quot len i addr
        FETCH         # base quot len i elem

        # Execute quotation
        4 PICK EXEC   # base quot len i

        1 ADD         # Increment i
    }

    DROP DROP DROP DROP
}
```

**Reduction:**
```ouro
PROC ARRAY_SUM ( base -- sum ) {
    DUP FETCH         # base len
    0                 # base len sum
    0                 # base len sum i

    WHILE { 3 PICK GT } {
        # sum += array[i]
        3 PICK        # base len sum i base
        OVER          # base len sum i base i
        ADD 1 ADD     # base len sum i addr
        FETCH         # base len sum i elem
        ROT ADD       # base len i sum'
        SWAP 1 ADD    # base len sum' i'
    }

    NIP NIP NIP       # sum
}
```

### 2.3 Multi-Dimensional Arrays

For 2D arrays (matrices), two conventions exist:

**Row-major ordering:**
$$\text{index}(i, j) = i \times \text{cols} + j$$

**Column-major ordering:**
$$\text{index}(i, j) = j \times \text{rows} + i$$

```ouro
# 3x3 matrix at base 200
# Header: rows, cols at 200, 201
# Data at 202+

PROC MATRIX_GET ( base i j -- val ) {
    # Get dimensions
    2 PICK FETCH      # rows
    2 PICK 1 ADD FETCH  # cols

    # index = i * cols + j + 2
    SWAP 4 ROLL MUL   # j + (i * cols)
    ADD 2 ADD         # offset

    # base + offset
    ROT ADD FETCH
}

# Create identity matrix
PROC IDENTITY_3x3 ( base -- ) {
    DUP 3 STORE       # rows = 3
    DUP 1 ADD 3 STORE # cols = 3

    # Diagonal entries
    DUP 2 ADD 1 STORE   # M[0,0] = 1
    DUP 6 ADD 1 STORE   # M[1,1] = 1
    DUP 10 ADD 1 STORE  # M[2,2] = 1

    DROP
}
```

### 2.4 Vectors as Typed Arrays

A vector interpretation layer:

```ouro
# Vector operations assuming array representation

PROC VEC_DOT ( base1 base2 -- dot ) {
    # Assumes equal length vectors
    OVER FETCH        # base1 base2 len
    0                 # base1 base2 len sum
    0                 # base1 base2 len sum i

    WHILE { 2 PICK GT } {
        # a[i] * b[i]
        4 PICK OVER ADD 1 ADD FETCH  # base1[i]
        4 PICK OVER ADD 1 ADD FETCH  # base2[i]
        MUL

        # sum += product
        ROT ADD SWAP
        1 ADD
    }

    DROP NIP NIP NIP
}

PROC VEC_ADD ( base1 base2 dest -- ) {
    # dest = base1 + base2 (element-wise)
    2 PICK FETCH      # len
    0                 # i

    WHILE { 2DUP GT } {
        # dest[i] = base1[i] + base2[i]
        4 PICK OVER ADD 1 ADD FETCH
        4 PICK OVER ADD 1 ADD FETCH
        ADD

        3 PICK OVER ADD 1 ADD STORE
        1 ADD
    }

    # Copy length to dest
    5 PICK FETCH
    4 PICK STORE

    DROP DROP DROP DROP DROP
}
```

---

## Part III: Linked Structures

### 3.1 The Linked List

A linked list node:

```
Node := (value, next_address)
```

Where `next_address = 0` indicates the end of the list.

```ouro
# Linked list operations

PROC LIST_PREPEND ( head_addr value -- ) {
    # Allocate node (requires allocator)
    ALLOC 2           # node_addr

    # Store value
    DUP ROT STORE     # value at node

    # Store old head as next
    SWAP FETCH        # old_head
    OVER 1 ADD STORE  # next = old_head

    # Update head pointer
    SWAP STORE
}

PROC LIST_TRAVERSE ( head_addr quot -- ) {
    FETCH             # current

    WHILE { DUP 0 NE } {
        DUP FETCH     # current value
        OVER EXEC     # apply quotation
        1 ADD FETCH   # current = current.next
    }

    DROP DROP
}
```

### 3.2 Doubly-Linked Lists

Node structure: `(prev, value, next)`

```ouro
# Node: 3 cells
# [0] = prev pointer
# [1] = value
# [2] = next pointer

PROC DLL_INSERT_AFTER ( node_addr value -- ) {
    # Allocate new node
    ALLOC 3           # new_addr

    # Set value
    DUP 1 ADD ROT STORE

    # new.prev = node
    DUP 2 PICK STORE

    # new.next = node.next
    OVER 2 ADD FETCH  # node.next
    OVER 2 ADD STORE  # new.next = node.next

    # node.next.prev = new (if exists)
    DUP 2 ADD FETCH   # new.next
    DUP 0 NE IF {
        OVER STORE    # new.next.prev = new
    } { DROP }

    # node.next = new
    SWAP 2 ADD STORE
}
```

### 3.3 The Absence of Garbage Collection

Linked structures in Ourochronos face a fundamental challenge: **manual memory management**.

Without garbage collection:
- Nodes must be explicitly freed
- Memory leaks are possible and undetectable
- Use-after-free creates silent corruption
- Reference cycles prevent cleanup

This is the price of simplicity. A language that manages memory must track references, which requires:
1. Reference counting (overhead per pointer operation)
2. Tracing GC (stop-the-world or concurrent complexity)
3. Region-based allocation (restricted patterns)

Ourochronos chooses none of these, placing the burden on the programmer.

---

## Part IV: Trees

### 4.1 Binary Trees

Node structure: `(value, left, right)`

```ouro
# Binary tree node: 3 cells
PROC TREE_NODE ( val -- addr ) {
    ALLOC 3
    DUP ROT STORE     # value
    DUP 1 ADD 0 STORE # left = null
    DUP 2 ADD 0 STORE # right = null
}

# In-order traversal
PROC TREE_INORDER ( root quot -- ) {
    DUP 0 EQ IF { DROP DROP } {
        # Traverse left
        DUP 1 ADD FETCH   # left child
        OVER TREE_INORDER

        # Visit node
        DUP FETCH         # value
        OVER EXEC

        # Traverse right
        2 ADD FETCH       # right child
        TREE_INORDER
    }
}

# Binary search tree insert
PROC BST_INSERT ( root val -- root' ) {
    DUP 0 EQ IF {
        DROP TREE_NODE    # Create root if empty
    } {
        2DUP FETCH GT IF {
            # val < root.val: go left
            DUP 1 ADD FETCH  # left
            ROT BST_INSERT   # recurse
            SWAP 1 ADD STORE # root.left = result
        } {
            # val >= root.val: go right
            DUP 2 ADD FETCH  # right
            ROT BST_INSERT   # recurse
            SWAP 2 ADD STORE # root.right = result
        }
    }
}
```

### 4.2 N-ary Trees

For trees with variable children count:

```
Node := (value, child_count, children_array_ptr)
```

Or using first-child/next-sibling representation:

```
Node := (value, first_child, next_sibling)
```

The latter converts any tree to a binary tree structure, simplifying traversal.

### 4.3 Tries (Prefix Trees)

A trie for string lookup:

```ouro
# Trie node: (is_terminal, children[256])
# This is memory-intensive: 257 cells per node

PROC TRIE_INSERT ( root str_base -- ) {
    # Traverse/create path for each character
    OVER FETCH        # root len

    0 WHILE { 2DUP GT } {
        # Get character
        2 PICK OVER ADD 1 ADD FETCH  # char

        # Navigate to child
        4 PICK          # root
        OVER 1 ADD ADD  # root + char + 1 (child slot)
        DUP FETCH       # current child

        DUP 0 EQ IF {
            # Create child node
            DROP
            TRIE_NODE
            DUP ROT STORE  # Store in parent
        } {
            NIP
        }

        # Child becomes new root
        4 ROLL DROP
        SWAP 3 ROLL DROP
        SWAP

        1 ADD
    }

    # Mark terminal
    DROP DROP
    1 STORE           # is_terminal = true
    DROP
}
```

---

## Part V: Hash Tables

### 5.1 The Missing Abstraction

Hash tables are conspicuously absent from Ourochronos. They require:

1. **Hash function:** Maps keys to integers
2. **Collision resolution:** Open addressing or chaining
3. **Dynamic resizing:** To maintain performance

### 5.2 A Manual Implementation

```ouro
# Simple hash table with chaining
# Header: [capacity, size, buckets_ptr]
# Bucket: linked list of (key, value, next)

PROC HASH_STRING ( str_base -- hash ) {
    DUP FETCH         # len
    0                 # hash
    0                 # i

    WHILE { 2 PICK GT } {
        # hash = hash * 31 + char
        31 MUL
        3 PICK OVER ADD 1 ADD FETCH
        ADD
        1 ADD
    }

    DROP NIP
}

PROC HASH_TABLE_CREATE ( capacity -- ht ) {
    # Allocate header + buckets
    DUP 3 ADD ALLOC   # ht

    # Initialise
    DUP ROT STORE     # capacity
    DUP 1 ADD 0 STORE # size = 0
    DUP 2 ADD         # buckets start

    # Zero buckets
    OVER FETCH        # capacity
    0 WHILE { 2DUP GT } {
        3 PICK OVER ADD
        0 STORE
        1 ADD
    }

    DROP DROP DROP
}

PROC HASH_TABLE_PUT ( ht key value -- ) {
    # Compute bucket
    OVER HASH_STRING
    3 PICK FETCH MOD  # bucket_idx
    3 PICK 3 ADD ADD  # bucket_addr

    # Create node
    ALLOC 3
    DUP 3 ROLL STORE     # key
    DUP 1 ADD 3 ROLL STORE  # value

    # Link to bucket head
    OVER FETCH
    OVER 2 ADD STORE     # node.next = bucket
    SWAP STORE           # bucket = node

    # Increment size
    1 ADD FETCH
    1 ADD
    SWAP 1 ADD STORE
}

PROC HASH_TABLE_GET ( ht key -- value found? ) {
    # Compute bucket
    DUP HASH_STRING
    2 PICK FETCH MOD
    2 PICK 3 ADD ADD
    FETCH                # bucket head

    # Search chain
    WHILE { DUP 0 NE } {
        2DUP FETCH       # node.key
        STR_EQ IF {
            1 ADD FETCH  # value
            NIP NIP
            1            # found
            EXIT
        }
        2 ADD FETCH      # next
    }

    DROP DROP
    0 0                  # not found
}
```

### 5.3 Performance Implications

Without built-in hash tables:
- O(n) lookup for associations vs O(1) average
- Manual memory management for buckets
- No automatic resizing
- String hashing must be implemented

For temporal computation, this matters: hash tables enable efficient memoisation, which is essential for fixed-point caching.

---

## Part VI: Graphs

### 6.1 Representation Choices

**Adjacency Matrix:**
$$A[i][j] = \begin{cases} 1 & \text{if edge } (i,j) \text{ exists} \\ 0 & \text{otherwise} \end{cases}$$

Space: O(V²), prohibitive for sparse graphs.

**Adjacency List:**
Each vertex stores list of neighbours.

Space: O(V + E), efficient for sparse graphs.

**Edge List:**
List of (source, destination) pairs.

Space: O(E), simple yet slow for neighbour lookup.

### 6.2 Adjacency List Implementation

```ouro
# Graph: [vertex_count, vertex_array_ptr]
# Vertex: [edge_count, edges_array_ptr]
# Edge: destination vertex index

PROC GRAPH_CREATE ( n -- g ) {
    # Allocate graph header
    ALLOC 2
    DUP ROT STORE     # vertex_count

    # Allocate vertex array
    OVER ALLOC
    OVER 1 ADD STORE  # vertex_array_ptr

    # Initialise vertices
    DUP 1 ADD FETCH   # vertices
    OVER FETCH        # n
    0 WHILE { 2DUP GT } {
        # Create empty edge list for vertex i
        0 ALLOC 2     # vertex
        DUP 0 STORE   # edge_count = 0
        DUP 1 ADD 0 STORE  # edges_ptr = null

        4 PICK OVER ADD STORE
        DROP
        1 ADD
    }

    DROP DROP DROP
}

PROC GRAPH_ADD_EDGE ( g src dst -- ) {
    # Get source vertex
    OVER 1 ADD FETCH  # vertices
    ROT ADD FETCH     # vertex[src]

    # Increment edge count
    DUP FETCH 1 ADD
    OVER STORE

    # Reallocate edges array (simplified: always grow)
    DUP FETCH ALLOC   # new edges
    # (copy old edges, add new)
    OVER 1 ADD FETCH  # old edges
    2 PICK FETCH 1 SUB  # old count
    # ... copy loop ...

    # Add new edge
    ROT 1 SUB ADD     # edges[count-1]
    ROT STORE         # = dst
}

PROC GRAPH_BFS ( g start -- ) {
    # Requires queue implementation
    # Mark visited, enqueue start
    # While queue not empty:
    #   dequeue vertex
    #   process vertex
    #   enqueue unvisited neighbours
}
```

### 6.3 Why Graphs Matter for Temporal Computation

Temporal dependencies form a graph:

```
Epoch 0 → Epoch 1 → Epoch 2
    ↓         ↓
  Value A   Value B
    └─────────┘
        ↓
    Fixed Point
```

To analyse temporal programs, we need graph operations:
- **Cycle detection:** Find temporal loops
- **Topological sort:** Order computations
- **Reachability:** Trace causal chains
- **Shortest path:** Find minimum epoch distances

Without graph structures, these analyses must be performed externally.

---

## Part VII: Priority Queues and Heaps

### 7.1 Binary Heap

A heap stored in array form:

```
Parent(i) = (i-1) / 2
Left(i) = 2*i + 1
Right(i) = 2*i + 2
```

```ouro
# Min-heap operations
# Heap: [size, capacity, data...]

PROC HEAP_PARENT ( i -- p ) {
    1 SUB 2 DIV
}

PROC HEAP_LEFT ( i -- l ) {
    2 MUL 1 ADD
}

PROC HEAP_RIGHT ( i -- r ) {
    2 MUL 2 ADD
}

PROC HEAP_PUSH ( heap val -- ) {
    # Get current size
    DUP FETCH         # size

    # Store value at end
    DUP 2 ADD         # data offset
    3 PICK ADD        # data[size]
    ROT STORE

    # Increment size
    DUP FETCH 1 ADD
    OVER STORE

    # Bubble up
    DUP FETCH 1 SUB   # i = size - 1
    WHILE { DUP 0 GT } {
        DUP HEAP_PARENT

        # Compare with parent
        3 PICK 2 ADD 2 PICK ADD FETCH  # heap[i]
        3 PICK 2 ADD 2 PICK ADD FETCH  # heap[parent]

        LT IF {
            # Swap
            3 PICK 2 ADD 2 PICK ADD  # &heap[i]
            3 PICK 2 ADD 2 PICK ADD  # &heap[parent]
            2DUP FETCH SWAP FETCH
            ROT STORE SWAP STORE

            NIP  # i = parent
        } {
            DROP DROP EXIT
        }
    }

    DROP DROP
}

PROC HEAP_POP ( heap -- val ) {
    # Get min (root)
    DUP 2 ADD FETCH   # val = heap[0]

    # Move last to root
    OVER FETCH 1 SUB  # size - 1
    2 PICK 2 ADD ADD FETCH  # heap[size-1]
    2 PICK 2 ADD STORE      # heap[0] = last

    # Decrement size
    OVER FETCH 1 SUB
    2 PICK STORE

    # Bubble down
    0                 # i = 0
    WHILE {
        # Find smallest of i, left, right
        DUP HEAP_LEFT
        3 PICK FETCH LT IF {
            # Has left child
            DUP HEAP_LEFT
            4 PICK 2 ADD 2 PICK ADD FETCH  # heap[left]
            4 PICK 2 ADD 3 PICK ADD FETCH  # heap[i]
            LT IF {
                # left is smaller
                DUP HEAP_LEFT
            } {
                DUP  # i is smallest so far
            }

            # Check right
            OVER HEAP_RIGHT
            5 PICK FETCH LT IF {
                5 PICK 2 ADD OVER ADD FETCH  # heap[right]
                5 PICK 2 ADD 2 PICK ADD FETCH  # heap[smallest]
                LT IF {
                    DROP 2 PICK HEAP_RIGHT  # right is smallest
                }
            }

            # If smallest != i, swap and continue
            2DUP NE IF {
                # Swap heap[i] and heap[smallest]
                # ... swap code ...
                NIP  # i = smallest
            } {
                DROP DROP EXIT
            }
        } {
            DROP EXIT  # No children
        }
    }

    DROP NIP
}
```

### 7.2 Applications in Temporal Computation

Priority queues enable:
- **Event scheduling:** Process epochs in order
- **Dijkstra's algorithm:** Shortest temporal paths
- **Merge operations:** Combine sorted temporal sequences
- **Best-first search:** Explore most promising timelines

---

## Part VIII: The Absence Catalogue

### 8.1 What Ourochronos Lacks

| Structure | Status | Impact |
|-----------|--------|--------|
| Dynamic arrays | Absent | Manual reallocation required |
| Hash tables | Absent | O(n) association lookup |
| Balanced trees | Absent | No guaranteed O(log n) operations |
| Graphs | Absent | No dependency analysis |
| Priority queues | Absent | No efficient scheduling |
| Strings (native) | Partial | Array-based, no Unicode |
| Sets | Absent | Must use hash or tree simulation |
| Deques | Absent | Must use doubly-linked lists |

### 8.2 What This Means

For **general programming:** Ourochronos requires significant boilerplate for common operations that other languages provide built-in.

For **temporal computation:** The lack of efficient structures impedes:
- Memoisation (no hash tables)
- Dependency tracking (no graphs)
- Event ordering (no priority queues)
- State snapshots (no persistent structures)

### 8.3 The Design Space

Ourochronos could address these gaps via:

1. **Built-in structures:** Add hash tables, vectors, etc. to the runtime
2. **Standard library:** Provide pure-Ourochronos implementations
3. **Foreign function interface:** Link to external libraries
4. **Metaprogramming:** Generate structure code from descriptions

Each choice has tradeoffs between complexity, performance, and purity.

---

## Part IX: Recommendations for Temporal Data Structures

### 9.1 Persistent Data Structures

For temporal computation, **persistent** (immutable) data structures offer advantages:

- **Time travel:** Access any historical version
- **Sharing:** Structural sharing reduces copying
- **Safety:** No aliasing problems

```ouro
# Persistent vector via path copying
# Each modification creates new version, shares unchanged parts

PROC PVEC_SET ( vec idx val -- vec' ) {
    # Copy spine, share leaves
    # Returns new vector root
}
```

### 9.2 Temporal-Aware Structures

Structures that understand epochs:

```ouro
# Temporal map: maps keys to values with epoch validity

PROC TMAP_PUT ( tmap key val epoch -- ) {
    # Store (key -> (val, epoch))
}

PROC TMAP_GET ( tmap key epoch -- val ) {
    # Find value valid at epoch
    # Navigate temporal versions
}
```

### 9.3 Causal Graphs

First-class support for dependency tracking:

```ouro
# Causal graph tracks value dependencies
CGRAPH_CREATE       # -- graph
CGRAPH_ADD_NODE val epoch  # graph -- graph node
CGRAPH_ADD_EDGE from to    # graph -- graph
CGRAPH_ANCESTORS node      # graph -- [ancestors]
CGRAPH_DETECT_CYCLE        # graph -- cycle?
```

---

## Conclusion: Structure and Time

Data structures are the scaffolding upon which computation is built. Ourochronos, in its minimalism, provides only the most basic scaffolding: memory and stack. This is philosophically coherent with its emphasis on temporal primitives over spatial complexity.

Yet temporal computation has structural needs. Dependency graphs, epoch-indexed maps, and causal tracking constitute necessities for reasoning about time-travelling programs.

The path forward requires providing the essential structures that make temporal computation tractable whilst preserving minimalism. A hash table serves as the foundation of efficient memoisation. A graph provides the natural representation of causal structure.

The structures we add define the thoughts we can think. Choose wisely.

---

*"In the end, we retain from our studies only that which we practically apply."*
*– Johann Wolfgang von Goethe*