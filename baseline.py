# import necessary libraries and modules
from vis_nav_game import Player, Action, Phase
import pygame
import cv2

import time
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import BallTree
import math
import heapq


class Marker(pygame.sprite.Sprite):
    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y
    

# Define a class for a player controlled by keyboard input using pygame
class KeyboardPlayerPyGame(Player):
    def __init__(self):
        # Initialize class variables
        self.fpv = None  # First-person view image
        self.last_act = Action.IDLE  # Last action taken by the player
        self.screen = None  # Pygame screen
        self.keymap = None  # Mapping of keyboard keys to actions
        super(KeyboardPlayerPyGame, self).__init__()
        
        # Variables for saving data
        self.count = 0  # Counter for saving images
        self.save_dir = "data/images/"  # Directory to save images to

        # Initialize SIFT detector
        # SIFT stands for Scale-Invariant Feature Transform
        self.sift = cv2.SIFT_create()
        # Load pre-trained codebook for VLAD encoding
        # If you do not have this codebook comment the following line
        # You can explore the maze once and generate the codebook (refer line 181 onwards for more)
        self.codebook = pickle.load(open("codebook.pkl", "rb"))
        # Initialize database for storing VLAD descriptors of FPV
        self.database = []

        # Dictionary to store image coordinates
        self.coordbook = {}

        self.fig, self.ax = plt.subplot()
        self.ax.set_aspect('equal')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Indexed Coordinates Plot')
        plt.grid(True)

        # Position data
        self.x = 0
        self.y = 0
        self.angle = 0

        # Additional state data
        self.turning = 0
        self.orientation = 'N'

        self.save_enable = True
        self.explore_compute = False
        self.wall_check = True

        self.every_other = True

        self.tolerance = 16
        self.valid_pixel = True
        
    def reset(self):
        # Reset the player state
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.x = 0
        self.y = 0

        # Initialize pygame
        pygame.init()

        # Define key mappings for actions
        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            # pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT
        }

        self.turningmap = {
            pygame.K_w: 1,
            pygame.K_a: 2,
            pygame.K_s: 3,
            pygame.K_d: 4
        }

    def act(self):
        """
        Handle player actions based on keyboard input
        """
        for event in pygame.event.get():
            #  Quit if user closes window or presses escape
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT
            # Check if a key has been pressed
            if event.type == pygame.KEYDOWN:
                # Miscillaneous keybinds
                # Reset angle to 0
                if event.key == pygame.K_p:
                    self.angle = 0
                    print("Reset angle to 0")
                # Plot map during exploration
                if event.key == pygame.K_m:
                    self.plot_base_map()
                # Toggle image saving
                if event.key == pygame.K_r:
                    self.save_enable = not self.save_enable
                    if self.save_enable: print("Image saving enabled")
                    else: print("Image saving disabled") 
                # Create new game instance? on top of the current one which should never be done why would you want to do this
                if event.key == pygame.K_l:
                    vis_nav_game.play(the_player=KeyboardPlayerPyGame())
                    print("Created new game instance")
                # Toggle computing kmeans and ball tree during exploration
                if event.key == pygame.K_i:
                    self.explore_compute = not self.explore_compute
                    if self.explore_compute: print("Explore_compute enabled")
                    else: print("Explore_compute disabled")
                # Toggle wall checking
                if event.key == pygame.K_n:
                    self.wall_check = not self.wall_check
                    if self.wall_check: print("Wall_check enabled")
                    else: print("Wall_check disabled")
                # Check if the pressed key is in the keymap
                if event.key in self.keymap and not self.turning:
                    # If yes, bitwise OR the current action with the new one
                    # This allows for multiple actions to be combined into a single action
                    self.last_act |= self.keymap[event.key]
                # Special cases for forward button
                if event.key == pygame.K_UP and self._state[1] != Phase.NAVIGATION and not self.turning:
                    if self.valid_pixel:
                        self.last_act |= Action.FORWARD
                    else:
                        self.last_act &= Action.IDLE
                # If a key is pressed that is not mapped to an action, then display target images
                else:
                    self.show_target_images()
            # Check if a key has been released
            if event.type == pygame.KEYUP:
                # Check if the released key is in the keymap
                if event.key in self.keymap:
                    # If yes, bitwise XOR the current action with the new one
                    # This allows for updating the accumulated actions to reflect the current sate of the keyboard inputs accurately
                    self.last_act ^= self.keymap[event.key]
                if event.key == pygame.K_UP and self._state[1] != Phase.NAVIGATION and not self.turning:
                    self.last_act &= Action.IDLE
        return self.last_act

    def show_target_images(self):
        """
        Display front, right, back, and left views of target location in 2x2 grid manner
        """
        targets = self.get_target_images()

        # Return if the target is not set yet
        if targets is None or len(targets) <= 0:
            return

        # Create a 2x2 grid of the 4 views of target location
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])

        w, h = concat_img.shape[:2]
        
        color = (0, 0, 0)

        concat_img = cv2.line(concat_img, (int(h/2), 0), (int(h/2), w), color, 2)
        concat_img = cv2.line(concat_img, (0, int(w/2)), (h, int(w/2)), color, 2)

        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1

        cv2.putText(concat_img, 'Front View', (h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Right View', (int(h/2) + h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Back View', (h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Left View', (int(h/2) + h_offset, int(w/2) + w_offset), font, size, color, stroke, line)

        cv2.imshow(f'KeyboardPlayer:target_images', concat_img)
        cv2.waitKey(1)

    def set_target_images(self, images):
        """
        Set target images
        """
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        self.show_target_images()

    def display_img_from_id(self, id, window_name):
        """
        Display image from database based on its ID using OpenCV
        """
        path = self.save_dir + str(id) + ".jpg"
        img = cv2.imread(path)
        cv2.imshow(window_name, img)
        cv2.waitKey(1)

    def display_map(self, window_name):
        """
        Display image from database based on its ID using OpenCV
        """
        path = 'map.jpg'
        img = cv2.imread(path)
        if img is not None:
            cv2.imshow(window_name, img)
            cv2.waitKey(1)
        else:
            print('map failed to read')

    def compute_sift_features(self):
        """
        Compute SIFT features for images in the data directory
        """
        length = len(os.listdir(self.save_dir))
        sift_descriptors = list()
        for i in range(length):
            path = str(i) + ".jpg"
            img = cv2.imread(os.path.join(self.save_dir, path))
            # Pass the image to sift detector and get keypoints + descriptions
            # We only need the descriptors
            # These descriptors represent local features extracted from the image.
            _, des = self.sift.detectAndCompute(img, None)
            # Extend the sift_descriptors list with descriptors of the current image
            sift_descriptors.extend(des)
        return np.asarray(sift_descriptors)
    
    def get_VLAD(self, img):
        """
        Compute VLAD (Vector of Locally Aggregated Descriptors) descriptor for a given image
        """
        # We use a SIFT in combination with VLAD as a feature extractor as it offers several benefits
        # 1. SIFT features are invariant to scale and rotation changes in the image
        # 2. SIFT features are designed to capture local patterns which makes them more robust against noise
        # 3. VLAD aggregates local SIFT descriptors into a single compact representation for each image
        # 4. VLAD descriptors typically require less memory storage compared to storing the original set of SIFT
        # descriptors for each image. It is more practical for storing and retrieving large image databases efficicently.

        # Pass the image to sift detector and get keypoints + descriptions
        # Again we only need the descriptors
        _, des = self.sift.detectAndCompute(img, None)
        # We then predict the cluster labels using the pre-trained codebook
        # Each descriptor is assigned to a cluster, and the predicted cluster label is returned
        pred_labels = self.codebook.predict(des)
        # Get number of clusters that each descriptor belongs to
        centroids = self.codebook.cluster_centers_
        # Get the number of clusters from the codebook
        k = self.codebook.n_clusters
        VLAD_feature = np.zeros([k, des.shape[1]])

        # Loop over the clusters
        for i in range(k):
            # If the current cluster label matches the predicted one
            if np.sum(pred_labels == i) > 0:
                # Then, sum the residual vectors (difference between descriptors and cluster centroids)
                # for all the descriptors assigned to that clusters
                # axis=0 indicates summing along the rows (each row represents a descriptor)
                # This way we compute the VLAD vector for the current cluster i
                # This operation captures not only the presence of features but also their spatial distribution within the image
                VLAD_feature[i] = np.sum(des[pred_labels==i, :] - centroids[i], axis=0)
        VLAD_feature = VLAD_feature.flatten()
        # Apply power normalization to the VLAD feature vector
        # It takes the element-wise square root of the absolute values of the VLAD feature vector and then multiplies 
        # it by the element-wise sign of the VLAD feature vector
        # This makes the resulting descriptor robust to noice and variations in illumination which helps improve the 
        # robustness of VPR systems
        VLAD_feature = np.sign(VLAD_feature)*np.sqrt(np.abs(VLAD_feature))
        # Finally, the VLAD feature vector is normalized by dividing it by its L2 norm, ensuring that it has unit length
        VLAD_feature = VLAD_feature/np.linalg.norm(VLAD_feature)

        return VLAD_feature

    def get_neighbor(self, img):
        """
        Find the nearest neighbor in the database based on VLAD descriptor
        """
        # Get the VLAD feature of the image
        q_VLAD = self.get_VLAD(img).reshape(1, -1)
        # This function returns the index of the closest match of the provided VLAD feature from the database the tree was created
        # The '1' indicates the we want 1 nearest neighbor
        _, index = self.tree.query(q_VLAD, 1)
        return index[0][0]

    def pre_nav_compute(self):
        """
        Build BallTree for nearest neighbor search and find the goal ID
        """
        # If this function is called after the game has started
        if self.count > 0:
            # below 3 code lines to be run only once to generate the codebook
            # Compute sift features for images in the database
            if not self.explore_compute:
                sift_descriptors = self.compute_sift_features()

                # KMeans clustering algorithm is used to create a visual vocabulary, also known as a codebook,
                # from the computed SIFT descriptors.
                # n_clusters = 64: Specifies the number of clusters (visual words) to be created in the codebook. In this case, 64 clusters are being used.
                # init='k-means++': This specifies the method for initializing centroids. 'k-means++' is a smart initialization technique that selects initial 
                # cluster centers in a way that speeds up convergence.
                # n_init=10: Specifies the number of times the KMeans algorithm will be run with different initial centroid seeds. The final result will be 
                # the best output of n_init consecutive runs in terms of inertia (sum of squared distances).
                # The fit() method of KMeans is then called with sift_descriptors as input data. 
                # This fits the KMeans model to the SIFT descriptors, clustering them into n_clusters clusters based on their feature vectors

                # TODO: try tuning the function parameters for better performance
                codebook = MiniBatchKMeans(n_clusters = 30, init='k-means++', n_init=10, verbose=1, batch_size=256).fit(sift_descriptors)
                pickle.dump(codebook, open("codebook.pkl", "wb"))

                # Build a BallTree for fast nearest neighbor search
                # We create this tree to efficiently perform nearest neighbor searches later on which will help us navigate and reach the target location
                
                # TODO: try tuning the leaf size for better performance
                tree = BallTree(self.database, leaf_size=30)
                self.tree = tree

            # Get the neighbor nearest to the front view of the target image and set it as goal
            targets = self.get_target_images()
            index = self.get_neighbor(targets[0])
            self.goal = index
            self.plot_coordinates(self.goal)
            print(f'Goal ID: {self.goal}')

    def pre_navigation(self):
        """
        Computations to perform before entering navigation and after exiting exploration
        """
        super(KeyboardPlayerPyGame, self).pre_navigation()
        self.keymap[pygame.K_UP] = Action.FORWARD
        self.pre_nav_compute()
        
    def display_next_best_view(self):
        """
        Display the next best view based on the current first-person view
        """

        # TODO: could you write this function in a smarter way to not simply display the image that closely 
        # matches the current FPV but the image that can efficiently help you reach the target?

        # Get the neighbor of current FPV
        # In other words, get the image from the database that closely matches current FPV
        index = self.get_neighbor(self.fpv)
        # Display the image 5 frames ahead of the neighbor, so that next best view is not exactly same as current FPV
        self.display_img_from_id(index+5, f'Next Best View')
        # Display the next best view id along with the goal id to understand how close/far we are from the goal
        print(f'Next View ID: {index+5} || Goal ID: {self.goal}')

    def turn(self, dest):
        """
        Helper function to turn 90 degrees
        """
        if self.angle == dest:
            self.last_act &= Action.IDLE
            self.turning = 0
        elif (self.orientation == 'W' and self.turning == 3):
            self.angle -= 1
            self.last_act |= Action.LEFT
        elif self.angle < dest or (self.orientation == 'S' and self.turning == 2):
            self.angle += 1
            self.last_act |= Action.RIGHT
        elif self.angle > dest:
            self.angle -= 1
            self.last_act |= Action.LEFT

    def shortest_path(self, points, start, end, jump_threshold):
        """
        Find shortest path from a start point to end point
        """
        def euclidean_distance(p1, p2):
            return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        # Initialize the priority queue with the start point
        queue = [(0, start)]
        visited = set()
        distances = {point: float('inf') for point in points}
        distances[start] = 0

        # Store the predecessor of each point
        predecessors = {point: None for point in points}

        while queue:
            # Get the point with the smallest distance
            current_distance, current_point = heapq.heappop(queue)
            visited.add(current_point)

            if current_point == end:
                # Backtrack from the end to the start to get the path
                path = []
                while current_point is not None:
                    path.append(current_point)
                    current_point = predecessors[current_point]
                path.reverse()
                return distances[end], path

            for point in points:
                if point not in visited and euclidean_distance(current_point, point) <= jump_threshold:
                    distance = current_distance + euclidean_distance(current_point, point)
                    if distance < distances[point]:
                        distances[point] = distance
                        predecessors[point] = current_point
                        heapq.heappush(queue, (distance, point))

        return distances[end], []


    def plot_coordinates(self, highlight_index=None):
        """
        Plot coordinates measured from maze for navigation use
        """
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        distance, path = self.shortest_path(list(self.coordbook.values()), (0, 0), \
                                  self.coordbook[highlight_index], 3)
        print(path)
        for index, (x, y) in self.coordbook.items():
            color = 'bo' if (x, y) in path else 'gs'
            ax.plot(x, y, color, markersize=5)
            # ax.text(x, y, str(index), fontsize=12, color='black')
        highlight_coords = self.coordbook[highlight_index]
        ax.plot(highlight_coords[0], highlight_coords[1], 'ro', markersize=7)
        
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Indexed Coordinates Plot')
        plt.grid(True)
        plt.savefig('map.jpg')
    
    def plot_base_map(self):
        """
        Plot coordinates measured from maze for exploration use and displays map
        """
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        for index, (x, y) in self.coordbook.items():
            color = 'gs'
            ax.plot(x, y, color, markersize=5)
        ax.plot(self.x, self.y, 'bo', markersize=7)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Indexed Coordinates Plot')
        plt.grid(True)
        plt.savefig('base_map.jpg')
        path = 'base_map.jpg'
        img = cv2.imread(path)
        if img is not None:
            cv2.imshow('Base Map', img)
            cv2.waitKey(1)
        else:
            print('map failed to read')


    def see(self, fpv):
        """
        Set the first-person view input
        """

        # Return if fpv is not available
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv

        # If the pygame screen has not been initialized, initialize it with the size of the fpv image
        # This allows subsequent rendering of the first-person view image onto the pygame screen
        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        def convert_opencv_img_to_pygame(opencv_image):
            """
            Convert OpenCV images for Pygame.

            see https://blanktar.jp/blog/2016/01/pygame-draw-opencv-image.html
            """
            opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
            shape = opencv_image.shape[1::-1]  # (height,width,Number of colors) -> (width, height)
            pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')

            return pygame_image

        pygame.display.set_caption("KeyboardPlayer:fpv")

        # If game has started
        if self._state:
            # If in exploration stage
            if self._state[1] == Phase.EXPLORATION:

                # TODO: could you employ any technique to strategically perform exploration instead of random exploration
                # to improve performance (reach target location faster)?
                
                # Lets self.angle wrap around
                keys_1 = pygame.key.get_pressed()
                if self.angle > 74:
                    self.angle -= 148
                elif self.angle < -74:
                    self.angle += 148
                
                # Check current view if close to wall
                height, width, _ = fpv.shape
                pixels = []
                for i in range(-10, 10):
                    pixels.append(((width // 2) + i, height - 16))
                self.valid_pixel = True
                if self.wall_check:
                    for pixel in pixels:
                        b, g, r = fpv[pixel[1], pixel[0]]
                        self.valid_pixel = ((b >= 255 - self.tolerance and g >= 255 - self.tolerance and r >= 255 - self.tolerance) or ((b <= 221 + self.tolerance and b >= 221 - self.tolerance) and (g <= 185 + self.tolerance and g >= 185 - self.tolerance) and (r <= 166 + self.tolerance and r >= 166 - self.tolerance))) and self.valid_pixel

                if self.last_act == Action.FORWARD and not self.valid_pixel:
                    self.last_act &= Action.IDLE

                # Moving forward
                if keys_1[pygame.K_UP] and not self.turning:
                    if self.valid_pixel:
                        match self.orientation:
                            case 'N': self.y += 1
                            case 'W': self.x -= 1
                            case 'S': self.y -= 1
                            case 'E': self.x += 1
                    print(f"{self.x}, {self.y}")
                # Moving backward
                if keys_1[pygame.K_DOWN] and not self.turning:
                    match self.orientation:
                        case 'N': self.y -= 1
                        case 'W': self.x += 1
                        case 'S': self.y += 1
                        case 'E': self.x -= 1
                    print(f"{self.x}, {self.y}")
                # Turning right
                if keys_1[pygame.K_RIGHT]:
                    self.angle += 1
                    print(f"{self.angle}")
                # Turning left
                if keys_1[pygame.K_LEFT]:
                    self.angle -= 1
                    print(f"{self.angle}")
                # 90 degree turns only
                if self.turning == 0:
                    if keys_1[pygame.K_w]:
                        self.turning = 1
                        self.orientation = 'N'
                    elif keys_1[pygame.K_a]:
                        self.turning = 2
                        self.orientation = 'W'
                    elif keys_1[pygame.K_s]:
                        self.turning = 3
                        self.orientation = 'S'
                    elif keys_1[pygame.K_d]:
                        self.turning = 4
                        self.orientation = 'E'
                # Destination angles for 90 degree turning
                match self.turning:
                    case 1: self.turn(0)
                    case 2: self.turn(-37)
                    case 3: self.turn(74)
                    case 4: self.turn(37)
                    
                # Compute KMeans and BallTree
                if keys_1[pygame.K_u] and self.explore_compute:
                    sift_descriptors = self.compute_sift_features()

                    codebook = MiniBatchKMeans(n_clusters = 20, init='k-means++', n_init=10, verbose=1, batch_size=256).fit(sift_descriptors)
                    pickle.dump(codebook, open("codebook.pkl", "wb"))

                    tree = BallTree(self.database, leaf_size=10)
                    self.tree = tree


                # Get full absolute save path
                save_dir_full = os.path.join(os.getcwd(),self.save_dir)
                save_path = save_dir_full + str(self.count) + ".jpg"
                # Create path if it does not exist
                if not os.path.isdir(save_dir_full):
                    os.mkdir(save_dir_full)
                # Save current FPV
                if self.save_enable and self.every_other:
                    cv2.imwrite(save_path, fpv)

                    # Get VLAD embedding for current FPV and add it to the database
                    VLAD = self.get_VLAD(self.fpv)
                    self.database.append(VLAD)
                    self.coordbook[self.count] = (self.x, self.y)
                    self.ax.plot(self.x, self.y, 'ro', markersize=7)
                    self.count = self.count + 1

                self.every_other = not self.every_other


            # If in navigation stage
            elif self._state[1] == Phase.NAVIGATION:
                # TODO: could you do something else, something smarter than simply getting the image closest to the current FPV?
                
                # Key the state of the keys
                keys = pygame.key.get_pressed()
                # If 'q' key is pressed, then display the next best view based on the current FPV
                if keys[pygame.K_q]:
                    self.display_next_best_view()
                    self.display_map('Map')
                

        # Display the first-person view image on the pygame screen
        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()


if __name__ == "__main__":
    import vis_nav_game
    # Start the game with the KeyboardPlayerPyGame player
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())
