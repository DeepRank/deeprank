#################################################################################
##
## ADAPTED FROM NLAP HDF5 BROWSER
## https://github.com/nanophotonics/nplab/blob/master/nplab/ui/hdf5_browser.py
##
#################################################################################

import os
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QAbstractItemModel, QFile, QIODevice, QModelIndex, Qt
from PyQt5.QtWidgets import QApplication, QTreeView, QFrame
import h5py
import functools

class DummyHDF5Group(dict):
    def __init__(self,dictionary, attrs ={}, name="DummyHDF5Group"):
        super(DummyHDF5Group, self).__init__()
        self.attrs = attrs
        for key in dictionary:
            self[key] = dictionary[key]
        self.name = name
        self.basename = name

    file = None
    parent = None

class HDF5TreeItem(object):

    ''' 
    Item in an HDF5 Tree 
    Adapted from NLAP  https://github.com/nanophotonics/nplab/blob/master/nplab/ui/hdf5_browser.py
    '''

    def __init__(self,data_file,parent,name,row):

        """Create a new item for an HDF5 tree
        data_file : HDF5 data file
            This is the file (NB must be the top-level group) containing everything
        parent : HDF5TreeItem
            The parent of the current item
        name : string
            The name of the current item (should be parent.name plus an extra component)
        row : int
            The index of the current item in the parent's children.
        """

        self.data_file = data_file
        self.parent = parent
        self.name = name
        self.row = row

        if parent is not None:
            assert name.startswith(parent.name)
            assert name in data_file

    @property
    def basename(self):
        return os.path.basename(self.name)

    _has_children = None
    @property
    def has_children(self):
        if self._has_children is None:
            self._has_children = hasattr(self.data_file[self.name],"keys")
        return self._has_children

    _children = None
    @property
    def children(self):
        if self.has_children is False:
            return []
        if self._children is None:
            keys = list(self.data_file[self.name].keys())
            self._children = [HDF5TreeItem(self.data_file, self, self.name.rstrip("/") + "/" + k, i) for i, k in enumerate(keys)]
        return self._children

    def purge_children(self):
        """Empty the cached list of children"""
        try:
            if self._children is not None:
                for child in self._children:
                    child.purge_children() # We must delete them all the way down!
                    self._children.remove(child)
                    del child # Not sure if this is needed...
                self._children = None
            self._has_children = None
        except:
            print("{} failed to purge its children".format(self.name))

    @property
    def h5item(self):
        """The underlying HDF5 item for this tree item."""
        assert self.name in self.data_file, "Error, {} is no longer a valid HDF5 item".format(self.name)
        return self.data_file[self.name]

    def __del__(self):
        self.purge_children()

def print_tree(item, prefix=""):
    """Recursively print the HDF5 tree for debug purposes"""
    if len(prefix) > 16:
        return # recursion guard
    print(prefix + item.basename)
    if item.has_children:
        for child in item.children:
            print_tree(child, prefix + "  ")

class HDF5ItemModel(QtCore.QAbstractItemModel):
    """
    This model takes its data from an HDF5 Group for display in a tree.
    It loads the file as the tree is expanded for speed - in the future it might implement sanity checks to
    abort loading very long folders.
    Adapted from NLAP  https://github.com/nanophotonics/nplab/blob/master/nplab/ui/hdf5_browser.py
    """

    def __init__(self,data_group):

        super().__init__()
        self.root_item = None
        self.data_group = data_group

    _data_group = None
    @property
    def data_group(self,new_data_group):
        if self.root_item is not None:
            del self.root_item
        self._data_group = new_data_group
        self.root_item = HDF5TreeItem(new_data_group.file,None,new_data_group.name,0)

    @data_group.setter
    def data_group(self, new_data_group):
        """Set the data group represented by the model"""
        if self.root_item is not None:
            del self.root_item
        self._data_group = new_data_group
        self.root_item = HDF5TreeItem(new_data_group.file, None, new_data_group.name, 0)

    def _index_to_item(self,index):
        ''' return tthe HDF5Item for a given index'''
        if index.isValid():
            return index.internalPointer()
        else:
            return self.root_item


    def index(self,row,column,parent_index):
        """Return the index of the <row>th child of parent
        :type row: int
        :type column: int
        :type parent_index: QtCore.QModelIndex
        """
        try:
            parent = self._index_to_item(parent_index)
            return self.createIndex(row, column, parent.children[row])
        except:
            return QtCore.QModelIndex()

    def parent(self, index=None):
        """Find the index of the parent of the item at a given index."""
        try:
            parent = self._index_to_item(index).parent
            return self.createIndex(parent.row, 0, parent)
        except:
            # Something went wrong with finding the parent so return an invalid index
            return QtCore.QModelIndex()

    def flags(self, index):
        """Return flags telling Qt what to do with the item"""
        return QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled

    def data(self, index, role):
        """The data represented by this item."""
        if role == QtCore.Qt.DisplayRole:
            return self._index_to_item(index).basename
        else:
            return None

    def headerData(self, section, orientation, role=None):
        """Return the header names - an empty string here!"""
        return [""]


    def rowCount(self, index):
        """The number of rows exposed by the model"""
        try:
            item = self._index_to_item(index)
            assert item.has_children
            return len(item.children)
        except:
            # if it doesn't have keys, assume there are no children.
            return 0

    def hasChildren(self, index):
        """Whether or not this object has children"""
        return self._index_to_item(index).has_children


    def columnCount(self, index=None, *args, **kwargs):
        """Return the number of columns"""
        return 1

    def refresh_tree(self):
        """Reload the HDF5 tree, resetting the model
        This causes all cached HDF5 tree information to be deleted, and any views
        using this model will automatically reload.
        """
        self.beginResetModel()
        self.root_item.purge_children()
        self.endResetModel()


    def selected_h5item_from_view(self, treeview):
        """Given a treeview object, return the selection, as an HDF5 object, or a work-alike for multiple selection.
        If one item is selected, we will return the HDF5 group or dataset that is selected.  If multiple items are
        selected, we will return a dummy HDF5 group containing all selected items.
        """
        items = [self._index_to_item(index) for index in treeview.selectedIndexes()]
        if len(items) == 1:
            return items[0].h5item
        elif len(items) > 1:
            return DummyHDF5Group({item.name: item.h5item for item in items})
        else:
            return None

    def set_up_treeview(self, treeview):
        """Correctly configure a QTreeView to use this model.
        This will set the HDF5ItemModel as the tree's model (data source), and in the future
        may set up context menus, etc. as appropriate."""
        treeview.setModel(self) # Make the tree view use this object as its model
        # Set up a callback to allow us to customise the context menu
        treeview.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        treeview.customContextMenuRequested.connect(functools.partial(self.context_menu, treeview))
        # Allow multiple objects to be selected
        treeview.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)


    def context_menu(self, treeview, position):
        """Generate a right-click menu for the items"""
        menu = QtWidgets.QMenu()
        actions = {}

        for operation in ['Details','Load in PyMol','Load in VMD','Targets']:
            actions[operation] = menu.addAction(operation)
        action = menu.exec_(treeview.viewport().mapToGlobal(position))

        items = [self._index_to_item(index) for index in treeview.selectedIndexes()]
        if len(items)>1:
            return
        item = items[0]

        if action == actions['Details']:
            print(item.name)

        if action == actions['Targets']:
            mol_name = item.name.split('/')[1]
            keys = list(self.root_item.data_file[mol_name+'/targets/'].keys())
            print('\n==> Target Values Found for %s' %mol_name)
            print('-----------------------------------------')
            for k in keys:
                print('{:10s}'.format(k) + ' : ' + '{:10f}'.format(self.root_item.data_file[mol_name+'/targets/'+k].value))
            print('-----------------------------------------')

class HDF5TreeWidget(QtWidgets.QTreeView):

    """A TreeView for looking at an HDF5 tree"""
    def __init__(self, datafile, **kwargs):
        """Create a TreeView widget that views the contents of an HDF5 tree.
        Arguments:
            datafile : nplab.datafile.Group
            the HDF5 tree to show
        Additional keyword arguments are passed to the QTreeView constructor.
        You may want to include parent, for example."""
        QtWidgets.QTreeView.__init__(self, **kwargs)

        self.model = HDF5ItemModel(datafile)
        self.model.set_up_treeview(self)
        self.sizePolicy().setHorizontalStretch(0)


    def selected_h5item(self):
        """Return the current selection as an HDF5 item."""
        return self.model.selected_h5item_from_view(self)

    #def __del__(self):
    #    del self.model # is this needed?  I'm never sure...


class HDF5Browser(QtWidgets.QWidget):
    """A Qt Widget for browsing an HDF5 file and graphing the data.
    """

    def __init__(self, data_file, parent=None):
        super(HDF5Browser, self).__init__(parent)
        self.data_file = data_file

        self.treeWidget = HDF5TreeWidget(data_file,
                                         parent=self,
                                         )
        self.selection_model = self.treeWidget.selectionModel() 
        #self.selection_model.selectionChanged.connect(self.selection_changed)
        #self.viewer = QTreeView() 
        #self.viewer = QFrame()

        top = QFrame() 
        top.setFrameShape(QFrame.StyledPanel)

        bottom = QFrame()
        bottom.setFrameShape(QFrame.StyledPanel)
        self.viewer = QtWidgets.QSplitter(Qt.Vertical)
        self.viewer.resize(1000,0)
        self.viewer.addWidget(top)
        self.viewer.addWidget(bottom)

        #self.refresh_tree_button = QtWidgets.QPushButton() #Create a refresh button
        #self.refresh_tree_button.setText("Refresh Tree")
        
        #adding the refresh button
        self.treelayoutwidget = QtWidgets.QWidget()     #construct a widget which can then contain the refresh button and the tree
        self.treelayoutwidget.setLayout(QtWidgets.QVBoxLayout())
        self.treelayoutwidget.layout().addWidget(self.treeWidget)
        #self.treelayoutwidget.layout().addWidget(self.refresh_tree_button) 
        
        #self.refresh_tree_button.clicked.connect(self.treeWidget.model.refresh_tree)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(self.treelayoutwidget)       #Add newly constructed widget (treeview and button) to the splitter
        splitter.addWidget(self.viewer)
        self.setLayout(QtWidgets.QHBoxLayout())
        self.layout().addWidget(splitter)


    def sizeHint(self):
        return QtCore.QSize(1024,768)

    def selection_changed(self, selected, deselected):
        """Callback function to update the displayed item when the tree selection changes."""
        print(selected)
        # try:
        #     self.viewer.data = self.treeWidget.selected_h5item()
        #     df.set_current_group(self.treeWidget.selected_h5item())
        # except Exception as e:
        #     print(e, 'That could be corrupted')
            
    def __del__(self):
        pass  # self.data_file.close()

if __name__ == '__main__':
    
    import sys

    data_file = h5py.File('1ak4.hdf5','r')


    app = QApplication(sys.argv)
    
    # model = HDF5ItemModel(data_file)
    # view = QTreeView()
    # view.setModel(model)
    # view.setWindowTitle("Simple Tree Model")
    # view.show()
    # sys.exit(app.exec_())
    # data_file.close()
    
    ui = HDF5Browser(data_file)
    ui.show()
    sys.exit(app.exec_())
    data_file.close()    