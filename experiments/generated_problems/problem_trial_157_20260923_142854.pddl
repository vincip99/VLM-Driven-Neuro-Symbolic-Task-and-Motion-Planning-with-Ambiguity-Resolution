(define (problem manipulation-task)
  (:domain manipulation)
  (:objects
    red_can - obj
    yellow_cube - obj
    brown_sorting_bin - location
    pot - location
  )
  (:init
    (on-table red_can)
    (on-table yellow_cube)
  )
  (:goal (and
    (on red_can brown_sorting_bin)
    (on yellow_cube pot)
    (on red_can brown_sorting_bin)
  ))
)
