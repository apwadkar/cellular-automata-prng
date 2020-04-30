import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def genSeed(l, c, i, r):
    """
    Generate an 8 byte seed from the conditions given

    Parameters
    ----------
    
    `l`: Size of automaton

    `c`: Column to extract generated values from

    `i`: Initial state of automaton encoded as a number

    `r`: Rule number encoded as a number

    Returns
    -------

    8 byte integer encoded as: `LLLLLLCC CCCCIIII IIIIIIII IIIIIIII IIIIIIII IIIIIIII IIIIIIII RRRRRRRR`
    """
    # Encoding: 6L 6C 46I 8R
    return ((l & 0x3F) << 0x3A) | ((c & 0x3F) << 0x34) | ((i & 0xFFFFFFFFF) << 8) | (r & 0xFF)

class PRNG:
    """
    Pseudo-random number generator based on simulating a 1-dimensional cellular automaton

    Initial values are set from an 8-byte integer seed

    See Also
    --------

    `genSeed`: Generates an 8-byte seed from desired configuration of automaton
    """
    u = np.array([[4], [2], [1]])

    def __init__(self, seed):
        self.setSeed(seed)

    def setSeed(self, newSeed):
        """
        Sets seed of PRNG and extracts automaton values from seed
        """
        self.seed = newSeed
        self.length = (self.seed >> 0x3A) & 0x3F
        if self.length > 44:
            print("Invalid length! Setting length to 44.")
            self.length = 44
        self.column = (self.seed >> 0x34) & 0x3F
        if self.column >= self.length:
            print("Invalid column! Setting column to end.")
            self.column = self.length - 1
        self.initial = (self.seed >> 8) & 0xFFFFFFFFF
        self.rule = self.seed & 0xFF
        self.__setIC()
    
    def __setIC(self):
        """
        Set initial conditions from the extracted seed values
        """
        self.imIndex = 0
        self.lastRow = np.array(
            [int(_) for _ in np.binary_repr(self.initial, self.length)],
            dtype=np.int8
        )
        self.ruleArr = np.array(
            [int(_) for _ in np.binary_repr(self.rule, 8)],
            dtype=np.int8)

    def __step(self, current):
        """
        Generate a new step given the current generation
        """
        # Get the neighbors and current cell
        y = np.vstack((np.roll(current, 1), current,
                       np.roll(current, -1))).astype(np.int8)
        # Get the result after applying the rule
        z = np.sum(y * self.u, axis=0).astype(np.int8)
        return self.ruleArr[7 - z]
    
    def generate(self, size, outputToFile=False):
        """
        Generate `size` bits simulated by a 1-dimensional cellular automaton
        """
        # Create matrix representing generations
        x = np.zeros((size, self.length), dtype=np.int8)
        # Set the initial state to the last computed state (on start, initial state from seed)
        x[0, :] = self.lastRow
        # Go through each generation
        for i in range(size - 1):
            x[i + 1, :] = self.__step(x[i, :])
        # Get the column (from seed) from the matrix and turn it into a number from the binary representation
        col = x[:, self.column]
        num = col.dot(2**np.arange(col.size)[::-1])
        # Output cellular automaton generations as an image
        if outputToFile:
            plt.imshow(x, cmap=plt.cm.binary)
            plt.xlabel("Column")
            plt.ylabel("Generations")
            initNum = self.lastRow.dot(2**np.arange(self.lastRow.size)[::-1])
            formatStr = f'{{:0{self.length}b}}'
            plt.title(f"Cellular Automaton (Seed: {self.seed:x})\nInitial State: {formatStr}".format(initNum))
            if not os.path.exists("{:x}".format(self.seed)):
                os.makedirs("{:x}".format(self.seed))
            plt.savefig("{:x}/ca_{:x}_{:x}".format(self.seed, self.imIndex, num))
        self.imIndex += 1
        # Set the last computed row to the last row of the matrix
        self.lastRow = x[size - 1, :]
        return num

    def generate8(self, outputToFile=False):
        """
        Generate `8` bits simulated by a 1-dimensional cellular automaton

        Alias of `self.generate(8)`
        """
        return self.generate(8, outputToFile)
    
    def generate16(self, outputToFile=False):
        """
        Generate `16` bits simulated by a 1-dimensional cellular automaton

        Alias of `self.generate(16)`
        """
        return self.generate(16, outputToFile)
    
    def generateFloat(self, outputToFile=False):
        """
        Generate a decimal value by normalizing a 16-bit generated value
        """
        return self.generate16(outputToFile) / float(2**16 - 1)
    
    def __repr__(self):
        return """    Seed: 0x{:x}
    Length: {}
    Column: {}
    Initial State: 0b{:b}
    Rule: {}""" \
        .format(self.seed, self.length, self.column, self.initial, self.rule)

if __name__ == '__main__':
    seed = genSeed(8, 3, 0b00110110, 30)
    gen = PRNG(seed)
    print("{}".format(gen))
    num = gen.generate(8, outputToFile=True)
    print(num)
    num = gen.generate(16, outputToFile=True)
    print(num)
    num = gen.generate(32, outputToFile=True)
    print(num)
    num = gen.generate(64, outputToFile=True)
    print(num)
    num = gen.generate(128, outputToFile=True)
    print(num)

    seed = genSeed(44, 20, 0b00110110001101100011011000110110001101100011, 30)
    gen.setSeed(seed)
    print("{}".format(gen))
    num = gen.generate(8, outputToFile=True)
    print(num)
    num = gen.generate(16, outputToFile=True)
    print(num)
    num = gen.generate(32, outputToFile=True)
    print(num)
    num = gen.generate(64, outputToFile=True)
    print(num)
    num = gen.generate(128, outputToFile=True)
    print(num)