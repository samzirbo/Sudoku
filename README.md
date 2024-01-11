# Sudoku

---
## Introduction
This project is a team effort developed for the Parallel and Distributed Programming course. The goal was to implement a solution for 
solving Sudoku puzzles using the pencil marking algorithm, inspired by **J. F. Crook's** paper _"A Pencil-and-Paper Algorithm for Solving Sudoku Puzzles."_

---
## Pencil Marking Algorithm
The implemented algorithm focuses on utilizing pencil marking techniques, specifically targeting singletons and hidden singles. 
The approach involves marking possible candidate values for each empty cell, gradually narrowing down the possibilities until a solution is reached.

---
## Performance Metrics
- **Parallelized Algorithm** - _using threads_ \
  \
The parallelization of the Sudoku solving algorithm became 'profitable' primarily for larger puzzles, specifically for the 49x49 size,
where the overhead of thread creation became negligible compared to the gains achieved through parallel processing. For smaller puzzles, the
increasing thread count led to longer execution times as the overhead of thread creation outweighed the benefits of parallelization. 

| Threads/Size |   9x9   |  16x16  |  49x49  |
|:------------:|:-------:|:-------:|:-------:|
|       1      |0.010s|0.037s|0.37s|
|       2      |0.012s|0.042s|0.31s|
|       4      |0.015s|0.056s|0.28s|
|       8      |0.022s|0.074s|0.27s|
  
- **Distributed Algorithm** - _using MPI_


| Threads/Size | 9x9 | 16x16 | 49x49 |
|--------------|-----|-------|-------|
| 1            |     |       |       |
| 2            |     |       |       |
| 4            |     |       |       |
| 8            |     |       |       |

---
## Contributors
- Ákos Péter
- Sam Zirbo
