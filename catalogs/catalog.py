import re
from my_jsbsim_env.catalogs.property import Property
from my_jsbsim_env.catalogs.jsbsim_catalog import JsbsimCatalog
from my_jsbsim_env.catalogs.my_catalog import MyCatalog


class DynamicCatalog(dict):
    """

    A class to store all jsbsim properties initiated and used during jsbsim simulation

    """

    def __getitem__(self, name):
        try:
            return super().__getitem__(name)   # 如果有这个属性，返回属性值
        except KeyError:  # look for the property in MyCatalog and JsbsimCatalog
            try:
                self[name] = MyCatalog[name].value
            except KeyError:
                self[name] = JsbsimCatalog[name].value
        return super().__getitem__(name)

    def __getattr__(self, name):  # 保证在没有这个属性值时能检索
        return self[name]

    def add_jsbsim_props(self, jsbsim_props):
        """

        Add to Catalog jsbsim properties from jbsbsim_props

        :param jsbsim_props: list of 'name_jsbsim (access)' of jsbsim properties

        """
        for jsbsim_prop in jsbsim_props:
            [name_jsbsim, access] = jsbsim_prop.split(" ")
            access = re.sub(r"[\(\)]", "", access)  # 去掉access中的括号
            name = re.sub(r"_$", "", re.sub(r"[\-/\]\[]+", "_", name_jsbsim))
            # get property name from jsbsim name

            if name not in self:
                try:
                    self[name] = JsbsimCatalog[name].value
                except KeyError:
                    self[name] = Property(name_jsbsim=name_jsbsim, access=access)


# 实例化 of DynamicCatalog used for simulation
Catalog = DynamicCatalog()
