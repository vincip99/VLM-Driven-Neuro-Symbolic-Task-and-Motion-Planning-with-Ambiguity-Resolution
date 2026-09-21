(define (problem manipulation-task)
  (:domain manipulation)
  (:objects
    red_can - obj
    yellow_cube - obj
    blue_can - obj
    pot - location
    sorting_bin - location
  )
  (:init
    (on-table red_can)
    (on-table yellow_cube)
    (on-table blue_can)
    (on-table pot)
    (on-table sorting_bin)
  )
  (:goal (and
    (on red_can sorting_bin)
    (on yellow_cube pot)
    (on blue_can sorting_bin)
  ))
)
