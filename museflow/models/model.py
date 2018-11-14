from cached_property import cached_property

from ..components.component import Component, using_scope

class Model(Component):

    @cached_property
    @using_scope
    def variables_initializer(self):
        if not self.built:
            raise RuntimeError("Attempt to access 'variables_initializer' before model is built")
        return tf.variables_initializer(tf.global_variables(self.variable_scope.name))
