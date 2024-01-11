import pandas as pd
import numpy as np
import threading
import time

def mapping(df):
    '''
    Given a dataframe with the sudoku board, it return the encoding & decoding mapping
        because all the values should be numbers to work efficiently with numpy arrays
    :param df: pandas DataFrame
    :return: dict, dict
    '''
    unique = set(np.unique(df.values.tolist()))
    unique.remove('0')
    encode = {'0': 0}
    decode = {0: '-'}
    for char in unique:
        encode[char] = len(encode)
        decode[len(decode)] = char
    return encode, decode

class Sudoku:
    def __init__(self, path, size=9, threads_no = 1):
        """
        Sudoku puzzle constructor that prepares the board and the candidates in order to be solved
        :param path: str
        :param size: int
        :param threads_no: int
        """
        # read the board from the file
        if size <= 9:
            matrix = np.genfromtxt(path, delimiter=',', filling_values=0, dtype=int)
            print("Initial board:")
            print(matrix)
        else:
            df = pd.read_csv(path, header=None).fillna(0)
            self.encode, self.decode = mapping(df)
            df = df.replace(self.encode)
            matrix = np.array(df.values.tolist(), dtype=int)
            print("Initial board:")
            print(df.replace(self.decode).to_string(index=False, header=False))

        # set the board and the candidates
        self.path = path
        self.board = matrix
        self.n = int(np.sqrt(self.board.shape[0]))
        self.candidates = {(i, j): self.board[i, j] if self.board[i, j] else set() for i in range(self.n ** 2) for j in range(self.n **2)}

        # prepare the threads
        self.threads = threads_no
        self.jobs = {}
        self.solutions = {}
        self.updated = []
        no_jobs = self.n ** 2 // self.threads

        for i in range(self.threads):
            self.solutions[i] = []
            self.jobs[i] = list(range(i * no_jobs, (i + 1) * no_jobs))
            if i == self.threads - 1:
                self.jobs[i] = list(range(i * no_jobs, self.n ** 2))

    # Utility methods
    def getRow(self, row, type_="values"):
        """
        Returns from the given row the requested type of values, where
            "indices" returns the indices of the empty cells
            "candidates" returns the candidates of the empty cells
            "values" returns the values of the row
        :param row: int
        :param type_: {"values", "indices", "candidates"}
        :return:  list or numpy array
        """
        indices = [(row, i) for i in range(self.n ** 2) if self.board[row, i] == 0]
        if type_ == "indices":
            return indices
        elif type_ == "candidates":
            return [candidate for index in indices for candidate in self.candidates[index]]
        return self.board[row, :]

    def getCol(self, col, type_="values"):
        """
        Returns from the given column the requested type of values, where
            "indices" returns the indices of the empty cells
            "candidates" returns the candidates of the empty cells
            "values" returns the values of the row
        :param col: int
        :param type_: {"values", "indices", "candidates"}
        :return:  list or numpy array
        """
        indices = [(i, col) for i in range(self.n ** 2) if self.board[i, col] == 0]
        if type_ == "indices":
            return indices
        elif type_ == "candidates":
            return [candidate for index in indices for candidate in self.candidates[index]]
        return self.board[:, col]

    def getBlock(self, block, type_="values"):
        """
        Returns from the given block the requested type of values, where
            "indices" returns the indices of the empty cells
            "candidates" returns the candidates of the empty cells
            "values" returns the values of the row
        :param block: int
        :param type_: {"values", "indices", "candidates"}
        :return:  list or numpy array
        """
        row = block // self.n * self.n
        col = block % self.n * self.n
        indices = [(row + i, col + j) for i in range(self.n) for j in range(self.n) if self.board[row + i, col + j] == 0]
        if type_ == "indices":
            return indices
        elif type_ == "candidates":
            return [candidate for index in indices for candidate in self.candidates[index]]
        return self.board[row:row+self.n, col:col+self.n].flatten()

    def setCell(self, row, col, value):
        """
        Sets the value of the given cell and updates its candidates to be its final value
        :param row: int
        :param col: int
        :param value: int
        :return: None
        """
        self.board[row, col] = value
        self.candidates[(row, col)] = value
        # print(f"C({row}, {col}) = {value}")

    # Sudoku methods for solving the puzzle
    def pencilMark(self, thread_id, rows):
        '''
        Pencil marks all the rows with possible candidates
        :param thread_id: int
        :param rows: list
        :return:
        '''
        for i in rows:
            for j in range(self.n ** 2):
                if self.board[i, j] != 0:
                    continue
                candidates = set(range(1, self.n**2 + 1))
                candidates -= set(np.unique(self.getRow(i)))
                candidates -= set(np.unique(self.getCol(j)))
                candidates -= set(np.unique(self.getBlock(i // self.n * self.n + j // self.n)))
                self.candidates[(i, j)] = candidates

    def getSingleton(self, thread_id, rows):
        '''
        Finds a singleton (empty cell that has only 1 candidate) in the given rows
            and stores it in the solutions dictionary for the given thread
        :param thread_id: int
        :param rows: list
        :return:
        '''
        for i in rows:
            for j in range(self.n ** 2):
                if self.board[i, j] != 0:
                    continue
                if len(self.candidates[(i, j)]) == 1:
                    self.solutions[thread_id].append((i, j))
                    return

    def getSingleHidden(self, thread_id, jobs):
        """
        Finds a hidden single (empty cell that is the only one in the row/colum/block that has a specific candidate)
                and stores it in the solutions dictionary for the given thread
        :param thread_id: int
        :param jobs: list
        :return: None
        """
        # check for single hidden in rows
        for row in jobs:
            row_candidates = self.getRow(row, "candidates")
            for _, column in self.getRow(row, "indices"):
                cell_candidates = self.candidates[(row, column)]
                for candidate in cell_candidates:
                    if row_candidates.count(candidate) == 1:
                        self.solutions[thread_id].append((row, column, candidate))
                        return

        # check for single hidden in columns
        for column in jobs:
            col_candidates = self.getCol(column, "candidates")
            for row, _ in self.getCol(column, "indices"):
                cell_candidates = self.candidates[(row, column)]
                for candidate in cell_candidates:
                    if col_candidates.count(candidate) == 1:
                        self.solutions[thread_id].append((row, column, candidate))
                        return

        # check for single hidden in blocks
        for block in jobs:
            block_candidates = self.getBlock(block, "candidates")
            for row, col in self.getBlock(block, "indices"):
                cell_candidates = self.candidates[(row, col)]
                for candidate in cell_candidates:
                    if block_candidates.count(candidate) == 1:
                        self.solutions[thread_id].append((row, col, candidate))
                        return

    def updateCandidates(self, i, j):
        """
        Updates the candidates of the neighbors of the given cell (row, col) because it has been filled
        :param i: int
        :param j: int
        :return:
        """
        value = self.board[i, j]
        block = i // self.n * self.n + j // self.n
        neighbors = set(self.getRow(i, "indices")) | set(self.getCol(j, "indices")) | set(self.getBlock(block, "indices"))
        for row, col in neighbors:
            if value in self.candidates[row, col]:
                self.candidates[row, col] -= {value}

    # Parallelizing methods
    def startThreads(self, worker):
        """
        Starts the threads with a given worker and waits for them to finish
        :param worker:
        :return: None
        """
        threads = []
        for i in range(self.threads):
            thread = threading.Thread(target=worker, args=(i, self.jobs[i],))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    def collectSolutions(self):
        """
        Collects the solutions from all the threads and resets the solutions
        :return: list
        """
        solutions = []
        for i in range(self.threads):
            solutions += self.solutions[i]
            self.solutions[i] = []
        return solutions

    # Main algorithm
    def solve(self):
        """
        Solves the sudoku puzzle:
            1. Pencil marks all the empty cells
            While there are singletons or hidden singles:
                2. Finds all the singletons and fills them
                3. If no singleton found, finds all the hidden singles and fills them
        :return:
        """

        # pencil mark
        self.startThreads(self.pencilMark)

        while True:
            # find the singletons
            self.startThreads(self.getSingleton)
            singletons = self.collectSolutions()
            if singletons:
                for solution in singletons:
                    row, col = solution
                    self.setCell(row, col, self.candidates[(row, col)].pop())
                for solution in singletons:
                    row, col = solution
                    self.updateCandidates(row, col)
            else:
                self.startThreads(self.getSingleHidden)
                hidden = self.collectSolutions()
                if hidden:
                    for solution in hidden:
                        row, col, candidate = solution
                        self.setCell(row, col, candidate)
                    for solution in hidden:
                        row, col, candidate = solution
                        self.updateCandidates(row, col)
                else:
                    break

        print(f"Final board: \n{self}")
        if self.isCorrect():
            print("\033[32mSolved!\033[0m")
        else:
            print("\033[31mNo solution found.\033[0m")

    # Final useful methods
    def isCorrect(self):
        """
        Checks if the board is correct i.e
            1. All the cells are filled
            2. All the rows, columns and blocks contain all the numbers from 1 to n^2
        :return: bool
        """
        full_set = set(range(1, self.n ** 2 + 1))
        if not self.board.all():
            return False
        for i in range(self.n ** 2):
            if set(self.getRow(i)) != full_set or set(self.getCol(i)) != full_set or set(self.getBlock(i)) != full_set:
                return False
        return True

    def saveSolution(self):
        """
        Given the path of the original board, it saves the solution in a new file
        :return: None
        """
        new_path = self.path.replace(".csv", "_solution.csv")
        df = pd.DataFrame(self.board)
        if self.n > 9:
            df.replace(self.decode, inplace=True)
        df.to_csv(new_path, index=False, header=False)

    def printCanidates(self):
        """
        Prints the candidates in a dataframe format for debugging purposes
        :return: None
        """
        df = pd.DataFrame(index=range(self.n ** 2), columns=range(self.n ** 2), dtype=object)
        for (row, col), candidates_set in self.candidates.items():
            df.iloc[row, col] = candidates_set
        print(df.to_string(index=True, col_space=4))

    def __str__(self):
        """
        Returns the board in a string format
        :return: str
        """
        df = pd.DataFrame(self.board)
        if self.n > 9:
            df.replace(self.decode, inplace=True)
        return df.to_string(index=False, header=False)

if __name__ == '__main__':
    threads = 8

    sudoku = Sudoku("16x16.csv", 16, threads)
    #########################################
    start = time.time()
    sudoku.solve()
    end = time.time()
    #########################################
    print(f"Time elapsed: {end - start:.5f}s")
    sudoku.saveSolution()

    # 9x9
    # 1 thread: 0.010s
    # 2 threads: 0.012s
    # 4 threads: 0.015s
    # 8 threads: 0.022s

    # 16x16
    # 1 thread: 0.037s
    # 2 threads: 0.042s
    # 4 threads: 0.056s
    # 8 threads: 0.074s

    #49x49
    # 1 thread: 0.37s
    # 2 threads: 0.31s
    # 4 threads: 0.28s
    # 8 threads: 0.27s





