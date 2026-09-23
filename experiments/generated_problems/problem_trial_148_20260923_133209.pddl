(define (problem manipulation-task)
  (:domain manipulation)
  (:objects
    red_red_can - obj
    yellow_yellow_cube - obj
    blue_blue_can - obj
    sorting_bin - location
    pot - location
  )
  (:init
    (on-table red_red_can)
    (on-table yellow_yellow_cube)
    (on-table blue_blue_can)
  )
  (:goal (and
    (on red_red_can sorting_bin)
    (on yellow_yellow_cube pot)
    (on blue_blue_can sorting_bin)
  ))
)
