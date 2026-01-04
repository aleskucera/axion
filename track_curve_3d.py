import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Track2D:
    """
    Standard 2D capsule/racetrack logic.
    Local Frame:
      - Origin (0,0) is center of rear circle.
      - X-axis points to center of front circle.
    """
    def __init__(self, r_rear, r_front, dist):
        self.r1 = r_rear
        self.r2 = r_front
        self.dist = dist

        # Centers in Local 2D
        self.c1 = np.array([0.0, 0.0])
        self.c2 = np.array([dist, 0.0])

        # --- Geometry Setup ---
        D_vec = self.c2 - self.c1
        L = np.linalg.norm(D_vec)
        
        # Calculate offset angle beta
        val = np.clip((self.r1 - self.r2) / L, -1.0, 1.0)
        beta = np.arccos(val)

        # Helper: Local 2D rotation
        def rotate(v, angle):
            c, s = np.cos(angle), np.sin(angle)
            return np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c])

        dir_vec = np.array([1.0, 0.0]) # Direction from c1 to c2 is always +X in local

        # Tangent Points (Local)
        n_top = rotate(dir_vec, beta)
        self.p1_top = self.c1 + self.r1 * n_top
        self.p2_top = self.c2 + self.r2 * n_top

        n_bot = rotate(dir_vec, -beta)
        self.p1_bot = self.c1 + self.r1 * n_bot
        self.p2_bot = self.c2 + self.r2 * n_bot

        # Lengths & Sweeps
        self.len_top = np.linalg.norm(self.p2_top - self.p1_top)
        self.len_bot = np.linalg.norm(self.p1_bot - self.p2_bot)

        # Front Arc (Clockwise from Top to Bot)
        ang_f_start = np.arctan2(n_top[1], n_top[0])
        ang_f_end = np.arctan2(n_bot[1], n_bot[0])
        self.sweep_front = (ang_f_start - ang_f_end) % (2 * np.pi)
        self.len_arc_front = self.sweep_front * self.r2

        # Rear Arc (Clockwise from Bot to Top)
        ang_r_start = np.arctan2(n_bot[1], n_bot[0])
        ang_r_end = np.arctan2(n_top[1], n_top[0])
        self.sweep_rear = (ang_r_start - ang_r_end) % (2 * np.pi)
        self.len_arc_rear = self.sweep_rear * self.r1

        self.ang_f_start = ang_f_start
        self.ang_r_start = ang_r_start

        # Accumulators
        self.L1 = self.len_top
        self.L2 = self.L1 + self.len_arc_front
        self.L3 = self.L2 + self.len_bot
        self.total_len = self.L3 + self.len_arc_rear

    def get_frame(self, u_in):
        u = u_in % self.total_len

        # 1. Top Line
        if u < self.L1:
            t = u / self.len_top
            pos = (1 - t) * self.p1_top + t * self.p2_top
            tan = (self.p2_top - self.p1_top) / self.len_top
            norm = np.array([-tan[1], tan[0]])
            return pos, tan, norm, 0.0

        # 2. Front Arc
        elif u < self.L2:
            arc_u = u - self.L1
            angle = self.ang_f_start - (arc_u / self.r2)
            pos = self.c2 + self.r2 * np.array([np.cos(angle), np.sin(angle)])
            tan = np.array([np.sin(angle), -np.cos(angle)])
            norm = np.array([np.cos(angle), np.sin(angle)])
            return pos, tan, norm, 1.0 / self.r2

        # 3. Bottom Line
        elif u < self.L3:
            line_u = u - self.L2
            t = line_u / self.len_bot
            pos = (1 - t) * self.p2_bot + t * self.p1_bot
            tan = (self.p1_bot - self.p2_bot) / self.len_bot
            norm = np.array([-tan[1], tan[0]])
            return pos, tan, norm, 0.0

        # 4. Rear Arc
        else:
            arc_u = u - self.L3
            angle = self.ang_r_start - (arc_u / self.r1)
            pos = self.c1 + self.r1 * np.array([np.cos(angle), np.sin(angle)])
            tan = np.array([np.sin(angle), -np.cos(angle)])
            norm = np.array([np.cos(angle), np.sin(angle)])
            return pos, tan, norm, 1.0 / self.r1

    def project(self, p):
        # p is array([x, y])
        candidates = []

        # 1. Top Line
        v = p - self.p1_top
        line_vec = self.p2_top - self.p1_top
        t = np.dot(v, line_vec) / np.dot(line_vec, line_vec)
        t = np.clip(t, 0.0, 1.0)
        closest = self.p1_top + t * line_vec
        dist_sq = np.sum((p - closest) ** 2)
        candidates.append((dist_sq, t * self.len_top))

        # 2. Front Arc
        v = p - self.c2
        ang_p = np.arctan2(v[1], v[0])
        diff = (self.ang_f_start - ang_p) % (2 * np.pi)
        if diff > self.sweep_front:
            if abs(diff - self.sweep_front) < abs(diff - 2 * np.pi):
                diff = self.sweep_front
            else:
                diff = 0.0
        dist_sq = np.sum((p - (self.c2 + self.r2 * np.array([np.cos(self.ang_f_start - diff), np.sin(self.ang_f_start - diff)])))**2)
        candidates.append((dist_sq, self.L1 + diff * self.r2))

        # 3. Bottom Line
        v = p - self.p2_bot
        line_vec = self.p1_bot - self.p2_bot
        t = np.dot(v, line_vec) / np.dot(line_vec, line_vec)
        t = np.clip(t, 0.0, 1.0)
        closest = self.p2_bot + t * line_vec
        dist_sq = np.sum((p - closest) ** 2)
        candidates.append((dist_sq, self.L2 + t * self.len_bot))

        # 4. Rear Arc
        v = p - self.c1
        ang_p = np.arctan2(v[1], v[0])
        diff = (self.ang_r_start - ang_p) % (2 * np.pi)
        if diff > self.sweep_rear:
            if abs(diff - self.sweep_rear) < abs(diff - 2 * np.pi):
                diff = self.sweep_rear
            else:
                diff = 0.0
        dist_sq = np.sum((p - (self.c1 + self.r1 * np.array([np.cos(self.ang_r_start - diff), np.sin(self.ang_r_start - diff)])))**2)
        candidates.append((dist_sq, self.L3 + diff * self.r1))

        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]


class Track3D:
    def __init__(self, r_rear, r_front, dist, origin, axis_x, axis_z):
        """
        r_rear, r_front, dist: Track shape parameters.
        origin: np.array([x, y, z]) - Center of the rear circle in 3D.
        axis_x: np.array([x, y, z]) - Vector pointing from Rear Center towards Front Center (defines track length direction).
        axis_z: np.array([x, y, z]) - Vector normal to the track plane.
        """
        self.track2d = Track2D(r_rear, r_front, dist)
        
        self.origin = np.array(origin, dtype=float)
        
        # Build orthonormal basis
        x = np.array(axis_x, dtype=float)
        x = x / np.linalg.norm(x)
        
        z = np.array(axis_z, dtype=float)
        z = z / np.linalg.norm(z)
        
        # y = z cross x
        y = np.cross(z, x)
        y = y / np.linalg.norm(y)
        
        # Re-orthogonalize x to ensure perfect 90 deg
        x = np.cross(y, z)
        x = x / np.linalg.norm(x)
        
        # Rotation Matrix (Local -> Global)
        # Columns are X, Y, Z axes
        self.R = np.column_stack((x, y, z))
        
        # Inverse Rotation (Global -> Local) = Transpose
        self.R_inv = self.R.T

    def get_frame_global(self, u):
        """
        Returns (pos, tan, norm, binorm) in Global 3D coords.
        """
        # 1. Get Local 2D frame
        pos2d, tan2d, norm2d, curvature = self.track2d.get_frame(u)
        
        # 2. Augment to 3D (z=0 in local frame)
        pos_local = np.array([pos2d[0], pos2d[1], 0.0])
        tan_local = np.array([tan2d[0], tan2d[1], 0.0])
        norm_local = np.array([norm2d[0], norm2d[1], 0.0])
        
        # 3. Rotate and Translate
        pos_global = self.origin + self.R @ pos_local
        tan_global = self.R @ tan_local
        norm_global = self.R @ norm_local
        binorm_global = np.cross(tan_global, norm_global) # Should align with Local Z
        
        return pos_global, tan_global, norm_global, binorm_global, curvature

    def project_global(self, point_3d):
        """
        Projects a 3D point onto the 3D curve.
        Returns: u_param, distance_to_curve_3d, projected_point_3d
        """
        # 1. Transform Global Point -> Local Frame
        # p_local = R_inv * (p_global - origin)
        diff = np.array(point_3d) - self.origin
        p_local_3d = self.R_inv @ diff
        
        # 2. Project onto 2D Track (ignoring local z)
        p_local_2d = p_local_3d[:2]
        u = self.track2d.project(p_local_2d)
        
        # 3. Get precise 2D point on track to calculate true distance
        pos2d, _, _, _ = self.track2d.get_frame(u)
        
        # 4. Construct Closest Point in 3D
        # The closest point is on the plane (local z=0)
        closest_local = np.array([pos2d[0], pos2d[1], 0.0])
        closest_global = self.origin + self.R @ closest_local
        
        dist = np.linalg.norm(point_3d - closest_global)
        
        return u, dist, closest_global

# --- Demo / Visualization ---
if __name__ == "__main__":
    # 1. Define Track Geometry
    origin = np.array([0.0, 0.0, 1.0])
    
    # Let's tilt the track! 
    # Pointing mostly +X, but tilted 30 deg up in Z
    axis_x = np.array([1.0, 0.0, 0.5]) 
    
    # Normal mostly +Z, but tilted back
    axis_z = np.array([-0.5, 0.0, 1.0])
    
    track3d = Track3D(r_rear=1.5, r_front=0.8, dist=5.0, 
                      origin=origin, axis_x=axis_x, axis_z=axis_z)

    # 2. Setup Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 3. Draw Track Path
    u_vals = np.linspace(0, track3d.track2d.total_len, 500)
    path_points = []
    for u in u_vals:
        pos, t, n, b, k = track3d.get_frame_global(u)
        path_points.append(pos)
    path_points = np.array(path_points)
    
    ax.plot(path_points[:,0], path_points[:,1], path_points[:,2], 'b-', lw=3, label="Track 3D")
    
    # 4. Draw Orientation Frames at intervals
    for u in np.linspace(0, track3d.track2d.total_len, 12, endpoint=False):
        pos, t, n, b, k = track3d.get_frame_global(u)
        # Tangent (Red)
        ax.quiver(pos[0], pos[1], pos[2], t[0], t[1], t[2], color='r', length=0.5, normalize=True)
        # Normal (Green)
        ax.quiver(pos[0], pos[1], pos[2], n[0], n[1], n[2], color='g', length=0.5, normalize=True)
        # Binormal (Blue - should point out of plane)
        ax.quiver(pos[0], pos[1], pos[2], b[0], b[1], b[2], color='b', length=0.5, normalize=True)

    # 5. Project Random Points
    np.random.seed(42)
    # Generate points in a box around the track
    random_points = (np.random.rand(10, 3) - 0.5) * 8.0 
    random_points[:,0] += 2.5 # Shift center
    
    for pt in random_points:
        u, dist, proj_pt = track3d.project_global(pt)
        
        # Draw Point
        ax.scatter(pt[0], pt[1], pt[2], color='orange', s=20)
        # Draw Projection Line
        ax.plot([pt[0], proj_pt[0]], [pt[1], proj_pt[1]], [pt[2], proj_pt[2]], 'k--', alpha=0.5)
        # Draw Projected Point
        ax.scatter(proj_pt[0], proj_pt[1], proj_pt[2], color='k', s=10)

    # Formatting
    ax.set_title("3D Track Projection Demo")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    # Equal aspect ratio hack for 3D
    # (Matplotlib 3D doesn't support 'equal' aspect directly well)
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])
    
    plt.legend()
    plt.show()
