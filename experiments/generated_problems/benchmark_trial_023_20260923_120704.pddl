(define (problem manipulation-task)
  (:domain manipulation)
  (:objects
    yellow_block - obj
    cooking_pot - location
  )
  (:init
    (on-table yellow_block)
  )
  (:goal (and
    (on yellow_block cooking_pot)
  ))
)
