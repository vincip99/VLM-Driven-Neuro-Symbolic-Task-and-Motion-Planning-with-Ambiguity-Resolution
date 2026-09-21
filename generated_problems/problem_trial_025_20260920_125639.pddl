(define (problem manipulation-task)
  (:domain manipulation)
  (:objects
    yellow_cube - obj
    blue_can - obj
    red_can - obj
    sorting_bin - location
    pot - location
  )
  (:init
    (on-table yellow_cube)
    (on-table blue_can)
    (on-table red_can)
  )
  (:goal (and
    (on red_can sorting_bin)
    (on yellow_cube pot)
    (on blue_can sorting_bin)
  ))
)
