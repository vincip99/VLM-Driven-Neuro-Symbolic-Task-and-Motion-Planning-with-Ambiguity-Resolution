(define (problem manipulation-task)
  (:domain manipulation)
  (:objects
    red_can - obj
    yellow_cube - obj
    blue_can - obj
    _table - location
    _pot - location
    _sorting_bin - location
  )
  (:init
    (on-table red_can)
    (on-table yellow_cube)
    (on-table blue_can)
  )
  (:goal (and
    (on red_can _sorting_bin)
    (on yellow_cube _pot)
    (on blue_can _table)
  ))
)
