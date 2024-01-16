import time

from mpi4py import MPI
from utils import Sudoku

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    sudoku = Sudoku("Puzzles/49x49.csv", 49, size)
else:
    sudoku = None

sudoku = comm.bcast(sudoku, root=0)
candidates = sudoku.pencilMark(rank)
candidates = comm.gather(candidates, root=0)

if rank == 0:
    asd = candidates
    for current_candidates in candidates:
        for key, value in current_candidates.items():
            sudoku.candidates[key] = value

    start_time = time.time()

    while True:
        for i in range(1, size):
            comm.send(sudoku, dest=i, tag=11)

        solution = sudoku.getSingleton(0)
        if solution == -1:
            solution = sudoku.getSingleHidden(0)
        # print(f"Rank {rank} found {solution}")
        if solution != -1:
            solutions = [solution]
        else:
            solutions = []

        for i in range(1, size):
            current_solution = comm.recv(source=i, tag=12)
            if current_solution != -1:
                solutions.append(current_solution)

        singletons = [sol for sol in solutions if len(sol) == 2]
        hidden = [sol for sol in solutions if len(sol) == 3]
        if len(singletons) > 0:
            for solution in singletons:
                row, col = solution
                sudoku.setCell(row, col, sudoku.candidates[(row, col)].pop())
            for solution in singletons:
                row, col = solution
                sudoku.updateCandidates(row, col)
        elif len(hidden) > 0:
            for solution in hidden:
                row, col, candidate = solution
                sudoku.setCell(row, col, candidate)
            for solution in hidden:
                row, col, candidate = solution
                sudoku.updateCandidates(row, col)
        else:
            sudoku.is_solved = True
            for i in range(1, size):
                comm.send(sudoku, dest=i, tag=11)
            break

    end_time = time.time()
    print(f"Final board: \n{sudoku}")
    if sudoku.isCorrect():
        print("\033[32mSolved!\033[0m")
    else:
        print("\033[31mNo solution found.\033[0m")



    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print(f"Elapsed time: {elapsed_time} seconds")

else:
    while True:
        sudoku = comm.recv(source=0, tag=11)
        if sudoku.is_solved:
            break
        solution = sudoku.getSingleton(rank)
        if solution == -1:
            solution = sudoku.getSingleHidden(rank)
        # print(f"Rank {rank} found {solution}")
        comm.send(solution, dest=0, tag=12)


