(define (problem p01)
  (:domain manipulation)
  (:objects 
    red_can - obj
    blue_can - obj
    green_can - obj
    yellow_cube - obj
    purple_cube - obj
    bin - location
    pot - location
  )
  (:init 
    (gripper-free)
    (on-table red_can)
  )
  (:goal 
    (on red_can bin)
  )
)