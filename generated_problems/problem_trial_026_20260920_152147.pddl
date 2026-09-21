(define (problem manipulation-task)
  (:domain manipulation)
  (:objects
    red_can - obj
    yellow_cube - obj
    blue_can - obj
    sorting_bin - location
    pot - location
  )
  (:init
    (on-table red_can)
    (on-table yellow_cube)
    (on-table blue_can)
    (on-table sorting_bin)
    (on-table pot)
  )
  (:goal (and
    (on red_can sorting_bin)
    (on yellow_cube pot)
    (on blue_can sorting_bin)
  ))
)
