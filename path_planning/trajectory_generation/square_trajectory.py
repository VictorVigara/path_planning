

def square_vertices(square_side, height):
        """
        Calculate the coordinates of all four vertices of a square given one vertex.
        
        Args:
            square_side (float): Length of one side of the square.
            height (float): Height of the square.
            
        Returns:
            tuple: A tuple containing four tuples, each representing the coordinates of one vertex.
        """
        # Calculate coordinates of other three vertices
        x1 = square_side
        y1 = 0.0
        x2 = square_side
        y2 = square_side
        x3 = 0.0
        y3 = square_side

        # Return the coordinates as a tuple of tuples
        return [[0.0, 0.0, height], [x1, y1, height], [x2, y2, height], [x3, y3, height]]