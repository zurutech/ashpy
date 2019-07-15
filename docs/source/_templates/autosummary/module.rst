{{ fullname }}
{{ underline }}

.. automodule:: {{ fullname }}

   {% block functions %}
   {% if functions %}
   .. rubric:: Functions

   .. autosummary::
      :nosignatures:
      :toctree: {{ name }}
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. rubric:: Classes

   .. autosummary::
      :nosignatures:
      :toctree: {{ name }}
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: Exceptions

   .. autosummary::
      :nosignatures:
      :toctree: {{ name }}
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
