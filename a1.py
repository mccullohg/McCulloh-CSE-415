# Assignment 1
# CSE 415, Sp 21
# Gordon McCulloh

import re


def is_multiple_of_11(n):
    """Return True if n is a multiple of 11; False otherwise."""
    if n % 11 == 0:  # remainder operator
        return True
    else:
        return False


def last_prime(m):
    """Return the largest prime number p that is less than or equal to m.
    You might wish to define a helper function for this.
    You may assume m is a positive integer."""
    # Check if m is prime and subtract 1 until a prime is found
    while is_prime(m) is False:
        m -= 1
    return m


# Helper function
def is_prime(m):
    """Return true if the input number, m, is prime."""
    if m <= 1:
        return False
    for ii in range(2, m):  # scanning all numbers leq half of m
        if m % ii == 0:  # divisibility check
            return False
    return True


def quadratic_roots(a, b, c):
    """Return the roots of a quadratic equation (real cases only).
    Return results in tuple-of-floats form, e.g., (-7.0, 3.0)
    Return "complex" if real roots do not exist."""
    # Calculate the discriminant
    d = b**2 - 4*a*c
    if d >= 0:  # real case - solutions to the quadratic equation
        root1 = (-b - d ** 0.5) / (2 * a)
        root2 = (-b + d ** 0.5) / (2 * a)
        return (root1, root2)
    else:
        return 'complex'


def perfect_shuffle(even_list):
    """Assume even_list is a list of an even number of elements.
    Return a new list that is the perfect-shuffle of the input.
    Perfect shuffle means splitting a list into two halves and then interleaving
    them. For example, the perfect shuffle of [0, 1, 2, 3, 4, 5, 6, 7] is
    [0, 4, 1, 5, 2, 6, 3, 7]."""
    L = len(even_list) // 2  # find half the length of the even list
    split1 = even_list[0:L]  # left hand
    split2 = even_list[L:]  # right hand
    shuffle = []  # initialize perfect shuffle
    for ii in range(L):
        shuffle = shuffle + [split1[ii], split2[ii]]  # concatenate list elements
    return shuffle


def five_times_list(input_list):
    """Assume a list of numbers is input. Using a list comprehension,
    return a new list in which each input element has been multiplied
    by 5."""
    return [5*ii for ii in input_list]  # automatically filter and multiply input


def triple_vowels(text):
    """Return a new version of text, with all the vowels tripled.
    For example:  "The *BIG BAD* wolf!" => "Theee *BIIIG BAAAD* wooolf!".
    For this exercise assume the vowels are
    the characters A,E,I,O, and U (and a,e,i,o, and u).
    Maintain the case of the characters."""
    vowels = 'AaEeIiOoUu'  # establish set of Upper/Lower vowels
    for letter in vowels:  # only iterate through vowels
        text = text.replace(letter, letter + letter + letter)  # string tool
    return text


def count_words(text):
    """Return a dictionary having the words in the text as keys,
    and the numbers of occurrences of the words as values.
    Assume a word is a substring of letters and digits and the characters
    '-', '+', '*', '/', '@', '#', '%', and "'" separated by whitespace,
    newlines, and/or punctuation (characters like . , ; ! ? & ( ) [ ] { } | : ).
    Convert all the letters to lower-case before the counting."""
    dictionary = {}  # initialize dictionary
    words = re.split(r'\s+|[~$^&[\](){}!;:.,?`"|_\\<>=]', text)  # split text into substrings with characters
    for word in words:  # iterate over each word in the text
        word = word.lower()  # make the word lower-case
        if word != '':  # evaluate words instead of whitespace
            if word in dictionary.keys():
                dictionary[word] = dictionary[word]+1  # count existing word
            else:
                dictionary[word] = 1  # add a new word
    return dictionary


def make_quartic_evaluator(a, b, c, d, e):
    """When called with 5 numbers, returns a function of one variable (x)
    that evaluates the quartic polynomial
    a x^4 + b x^3 + c x^2 + d x + e.
    For this exercise Your function definition for make_quartic_evaluator
    should contain a lambda expression."""
    return lambda x: a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e


class Polygon:
    """Polygon class."""

    def __init__(self, n_sides, lengths=None, angles=None):
        # Qualify the self parameter
        self.n_sides = n_sides
        self.lengths = lengths
        self.angles = angles

    def is_rectangle(self):
        """returns True if the polygon is a rectangle,
        False if it is definitely not a rectangle, and None
        if the angle list is unknown (None)."""
        # Disqualify polygons without 4 sides
        if self.n_sides != 4:
            return False
        # Check if all angles are equal to 90
        return self._angle_test(90)

    def is_rhombus(self):
        # Disqualify polygons without 4 sides
        if self.n_sides != 4:
            return False
        # Equilateral test
        return self._length_test()

    def is_square(self):
        # All squares are rectangles
        if self.is_rectangle() is False:
            return False
        # Equilateral and all angles are equal to 90
        return self._length_test() and self._angle_test(90)

    def is_regular_hexagon(self):
        # Disqualify polygons without 6 sides
        if self.n_sides != 6:
            return False
        # Equilateral false case
        if self._length_test() is False or self._angle_test(120) is False:
            return False
        # Equilateral and all angles are equal to 120
        return self._length_test() and self._angle_test(120)

    def is_isosceles_triangle(self):
        # Disqualify polygons without 3 sides
        if self.n_sides != 3:
            return False
        # Empty case
        if self.lengths is None and self.angles is None:
            return None
        # Length list nonempty
        if self.lengths is not None:
            for side in self.lengths:
                if self.lengths.count(side) >= 2:
                    return True
        # Angle list nonempty
        else:
            for angle in self.angles:
                if self.angles.count(angle) >= 2:
                    return True
        # Otherwise, not isosceles
        return False

    def is_equilateral_triangle(self):
        # Disqualify polygons without 3 sides
        if self.n_sides != 3:
            return False
        # Equilateral false case
        if self._length_test() is False:
            return False
        # Equilateral and all angles are equal to 60
        return self._length_test() or self._angle_test(60)

    def is_scalene_triangle(self):
        # Disqualify polygons without 3 sides
        if self.n_sides != 3:
            return False
        # If a triangle is not isosceles, it is scalene
        if self.is_isosceles_triangle() is not None:
            return not self.is_isosceles_triangle()
        else:
            return None

    # Helper function to see if all angles in the polygon match a desired angle
    def _angle_test(self, check_angle):
        # empty case
        if self.angles is None:
            return None
        # search angles against the check angle
        for angle in self.angles:
            if angle != check_angle:
                return False
            else:
                return True

    # Helper function to test whether the polygon is equilateral
    def _length_test(self):
        # empty case
        if self.lengths is None:
            return None
        side = self.lengths[0]  # parse the first side length
        # test if all side lengths are equal
        return self.lengths.count(side) == self.n_sides
