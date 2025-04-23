import math

class CFNode:
    """
    A minimal BIRCH-style CF node storing:
      - name: a label for debugging
      - N:    integer (count of points)
      - LS:   tuple (sum_x, sum_y)
      - SS:   tuple (sum_x^2, sum_y^2)
    """
    def __init__(self, N=0, LS=(0.0, 0.0), SS=(0.0, 0.0), name="CF"):
        self.name = name
        self.N = N
        self.LS = LS
        self.SS = SS

    def centroid(self):
        """Return the centroid (cx, cy) for this CF."""
        if self.N == 0:
            return (0.0, 0.0)
        return (self.LS[0] / self.N, self.LS[1] / self.N)

    def __str__(self):
        return f"{self.name} <{self.N}, {self.LS}, {self.SS}>"

def dist_coordwise(cf: CFNode, point: tuple) -> tuple:
    """Coordinate-wise distance between CF's centroid and 'point'."""
    cx, cy = cf.centroid()
    px, py = point
    return (abs(px - cx), abs(py - cy))

def within_boundary(dist_tuple, boundary=(1.5,1.5)) -> bool:
    """Check coordinate-wise if dist_tuple <= boundary."""
    return (dist_tuple[0] <= boundary[0]) and (dist_tuple[1] <= boundary[1])

def add_point_to_cf(cf: CFNode, point: tuple) -> None:
    """Update CF in place with a new point = (px, py)."""
    px, py = point
    cf.N += 1
    cf.LS = (cf.LS[0] + px, cf.LS[1] + py)
    cf.SS = (cf.SS[0] + px**2, cf.SS[1] + py**2)

def create_new_cf(point: tuple, name="CF_new") -> CFNode:
    """Return a new CF containing exactly one point."""
    px, py = point
    return CFNode(
        N=1,
        LS=(px, py),
        SS=(px**2, py**2),
        name=name
    )

def merge_cf(cfA: CFNode, cfB: CFNode, name="MergedCF") -> CFNode:
    """
    Merge CF A into CF B => returning a new CF with sums and N combined.
    """
    newN  = cfA.N + cfB.N
    newLS = (cfA.LS[0] + cfB.LS[0], cfA.LS[1] + cfB.LS[1])
    newSS = (cfA.SS[0] + cfB.SS[0], cfA.SS[1] + cfB.SS[1])
    return CFNode(newN, newLS, newSS, name=name)

def dist_cf_to_cf(cfA: CFNode, cfB: CFNode) -> tuple:
    """Coordinate-wise distance between centroids of two CFs."""
    cA = cfA.centroid()
    cB = cfB.centroid()
    return (abs(cA[0] - cB[0]), abs(cA[1] - cB[1]))

def describe_cf(cf: CFNode, indent=""):
    """Return a short multiline string describing the CF."""
    cx, cy = cf.centroid()
    return (f"{indent}{cf}\n"
            f"{indent}  Centroid: ({cx:.2f}, {cy:.2f})")


def run_birch_demo():
    boundary = (1.5, 1.5)  # same boundary for all scenarios

    print("=== Problem 1 scenario ===")
    # Suppose we have CF1 with 1 point
    CF1 = CFNode(N=1, LS=(5.0, 7.0), SS=(25.0, 49.0), name="CF1")
    print(f"Initial: {describe_cf(CF1)}")

    # x2 arrives => distance is effectively (0.5, 1.0) => within boundary
    x2 = (4.0, 6.0)
    dist2 = (0.5, 1.0)
    print(f"Inserting x2={x2} with dist={dist2} (boundary={boundary})...")
    if within_boundary(dist2, boundary):
        add_point_to_cf(CF1, x2)
        print(f"  => x2 added into CF1")
    else:
        CF2 = create_new_cf(x2, name="CF2")
        print(f"  => new CF2 created: {describe_cf(CF2)}")

    print(f"After Problem 1:\n{describe_cf(CF1)}\n")

    #====================================================
    print("=== Problem 2 scenario ===")
    # x2b arrives => distance = (2.5, 2.1) => out of boundary => new CF2
    x2b = (7.0, 8.0)
    dist2b = (2.5, 2.1)
    print(f"Inserting x2b={x2b} with dist={dist2b} (boundary={boundary})...")
    if within_boundary(dist2b, boundary):
        add_point_to_cf(CF1, x2b)
        print(f"  => x2b added to CF1")
    else:
        CF2 = create_new_cf(x2b, name="CF2")
        print(f"  => new CF2 created:\n{describe_cf(CF2)}")

    print(f"After Problem 2:\n{describe_cf(CF1)}\n")

    #====================================================
    print("=== Problem 3 scenario ===")
    # x3 => dist (3.5,4.1) => out of boundary => new CF3
    x3 = (10.0, 12.0)
    dist3 = (3.5, 4.1)
    print(f"Inserting x3={x3} with dist={dist3} (boundary={boundary})...")
    if within_boundary(dist3, boundary):
        add_point_to_cf(CF1, x3)
        print("  => x3 added to CF1")
    else:
        CF3 = create_new_cf(x3, "CF3")
        print(f"  => new CF3 created:\n{describe_cf(CF3)}")

    print("After Problem 3 so far:")
    print(f"{describe_cf(CF1)}")
    if 'CF2' in locals():
        print(describe_cf(CF2))
    if 'CF3' in locals():
        print(describe_cf(CF3))
    print()

    #====================================================
    print("=== Problem 4 scenario ===")
    # We have CF1 <2,...> and CF2 <3,...>, possibly merges, then x6 arrives.
    # For demonstration, let's override CF1, CF2 with new values:
    CF1 = CFNode(2, (9.0, 9.0), (81.0, 81.0), name="CF1")
    CF2 = CFNode(3, (12.0,15.0), (144.0,225.0), name="CF2")
    print("Current CF nodes:")
    print(describe_cf(CF1))
    print(describe_cf(CF2))

    # Suppose we decide CF1 is close to CF2 => merge them
    dist_cf1_cf2 = dist_cf_to_cf(CF1, CF2)
    # If that distance is small enough => we do a merge
    if within_boundary(dist_cf1_cf2, boundary=(2.0,2.0)):
        CF2_merged = merge_cf(CF1, CF2, name="CF2_merged")
        print(f"  => Merged CF1 into CF2 => {describe_cf(CF2_merged)}")
        # let CF1 go away, replaced by CF2_merged
        CF1 = None
        CF2 = CF2_merged
    else:
        print(f"  => No merge (dist_cf1_cf2={dist_cf1_cf2} is out of range)")

    # x6 arrives with dist to CF1=(4.7,4.9), dist to CF2=(0.7,0.5)
    x6 = (13.0,16.0)
    dist6_cf1 = (4.7,4.9)
    dist6_cf2 = (0.7,0.5)
    print(f"Now inserting x6={x6} => dist to CF1={dist6_cf1}, dist to CF2={dist6_cf2}")

    did_insert = False
    if CF1 is not None and within_boundary(dist6_cf1, boundary):
        add_point_to_cf(CF1, x6)
        print(f"  => x6 updated CF1 => {describe_cf(CF1)}")
        did_insert = True
    elif within_boundary(dist6_cf2, boundary):
        add_point_to_cf(CF2, x6)
        print(f"  => x6 updated CF2 => {describe_cf(CF2)}")
        did_insert = True
    else:
        # new CF
        CF6 = create_new_cf(x6, "CF6")
        print(f"  => x6 => created new CF6 =>\n{describe_cf(CF6)}")

    print("After Problem 4 so far:")
    if CF1 is not None:
        print(describe_cf(CF1))
    print(describe_cf(CF2))
    if 'CF6' in locals():
        print(describe_cf(CF6))
    print()

    #====================================================
    print("=== Problem 5 scenario ===")
    # CF1 <2,(c2,d2),(e2,f2)> & CF2 <3,(c5,d5),(e5,f5)>, x6 => distances = (5.2,4.8) or (2.7,3.5)
    CF1 = CFNode(2, (9.0,9.0), (81.0,81.0), "CF1")
    CF2 = CFNode(3, (12.0,15.0), (144.0,225.0), "CF2")
    x6p5 = (20.0,19.0)
    dist6p5_cf1 = (5.2,4.8)
    dist6p5_cf2 = (2.7,3.5)
    print(f"CF1 now: {describe_cf(CF1)}")
    print(f"CF2 now: {describe_cf(CF2)}")
    print(f"Inserting x6={x6p5} => dist(CF1)={dist6p5_cf1}, dist(CF2)={dist6p5_cf2}")

    # Possibly check if CF1<->CF2 should be merged first, or see if x6p5 fits
    dist_cf1_cf2b = dist_cf_to_cf(CF1, CF2)
    if within_boundary(dist_cf1_cf2b, boundary=(2.0,2.0)):
        M = merge_cf(CF1, CF2, name="CF2_merged")
        CF2 = M
        CF1 = None
        print(f"  => Merged CF1->CF2 => {describe_cf(CF2)}")

    inserted5 = False
    if CF1 is not None and within_boundary(dist6p5_cf1, boundary):
        add_point_to_cf(CF1, x6p5)
        print("  => x6 => updated CF1")
        inserted5 = True
    elif within_boundary(dist6p5_cf2, boundary):
        add_point_to_cf(CF2, x6p5)
        print("  => x6 => updated CF2")
        inserted5 = True
    else:
        # create new CF
        CF6p5 = create_new_cf(x6p5, "CF_new_6")
        print("  => x6 => new CF_new_6 created =>")
        print(describe_cf(CF6p5))

    print("\nAfter Problem 5 final structure:")
    if CF1 is not None:
        print(describe_cf(CF1))
    print(describe_cf(CF2))
    if 'CF6p5' in locals():
        print(describe_cf(CF6p5))
    print("End of demonstration.\n")

# Running the script:
if __name__ == "__main__":
    run_birch_demo()