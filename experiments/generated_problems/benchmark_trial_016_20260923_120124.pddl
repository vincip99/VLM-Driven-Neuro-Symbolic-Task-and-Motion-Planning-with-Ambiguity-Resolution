(define (problem manipulation-task)
  (:domain manipulation)
  (:objects
    red_can - obj
    blue_can - obj
    green_can - obj
    _sorting_bin - location
    _pot - location
  )
  (:init
    (on-table red_can)
    (on-table blue_can)
    (on-table green_can)
  )
  (:goal (and
    (on red_can _sorting_bin)
    (on blue_can _pot)
    (on green_can _sorting_bin)
  ))
)
