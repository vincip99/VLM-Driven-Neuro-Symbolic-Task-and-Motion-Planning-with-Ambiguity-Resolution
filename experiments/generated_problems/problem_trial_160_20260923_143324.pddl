(define (problem manipulation-task)
  (:domain manipulation)
  (:objects
    red_can - obj
    yellow_cube - obj
    sortbin1 - location
    pot - location
    blue_can - obj
    sortbin2 - location
  )
  (:init
    (on-table red_can)
    (on-table yellow_cube)
  )
  (:goal (and
    (on red_can sortbin1)
    (on yellow_cube pot)
    (on blue_can sortbin2)
  ))
)
