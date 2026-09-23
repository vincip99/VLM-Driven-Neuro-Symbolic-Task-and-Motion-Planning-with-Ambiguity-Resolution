(define (problem manipulation-task)
  (:domain manipulation)
  (:objects
    red_can - obj
    yellow_cube - obj
    blue_can - obj
    bottom_receptacle_(pot) - location
    top_receptacle_(sorting_bin) - location
  )
  (:init
    (on-table red_can)
    (on-table yellow_cube)
    (on-table blue_can)
  )
  (:goal (and
    (on red_can top_receptacle_(sorting_bin))
    (on yellow_cube bottom_receptacle_(pot))
    (on blue_can top_receptacle_(sorting_bin))
  ))
)
