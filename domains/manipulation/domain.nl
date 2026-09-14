The Manipulation domain involves a world with a table, a robot manipulator, and various objects (cups, bins, cylinders, cubes, mugs ecc.). The robot has a single gripper that can be used to pick up and drop objects. The goal is to move objects from their initial locations to the desired goal locations using the robot and its gripper.

The domain includes two actions:

2. pick: This action allows the robot to pick up an object using its gripper. The preconditions are that the object must be on the table and the gripper must be free (not holding any object). The effect is that the robot is now carrying the object with the gripper, the object is no longer on the table, and the gripper is no longer free.

3. place: This action allows a robot to drop an object it is carrying in a specific location using its gripper. The preconditions are that the robot must be carrying the object with the gripper. The effect is that the object is now in the specified location, the gripper is free, and the robot is no longer carrying the object with the gripper.